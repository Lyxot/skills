// hwgate-fan: pin every fan at its maximum for a --stable run, or hand them back to macOS.
//
//   hwgate-fan status   read-only; works as any user
//   hwgate-fan max      root: unlock manual mode (Ftst) and target each fan's F<n>Mx
//   hwgate-fan auto     root: return each fan to automatic and drop the unlock
//
// Only ever raises fan speed: there is no command that sets a target below the maximum.
#include <IOKit/IOKitLib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum { KERNEL_INDEX_SMC = 2, CMD_READ_BYTES = 5, CMD_WRITE_BYTES = 6, CMD_READ_KEYINFO = 9 };

typedef struct { char major, minor, build, reserved; uint16_t release; } Vers;
typedef struct { uint16_t version, length; uint32_t cpu, gpu, mem; } PLimit;
typedef struct { uint32_t size, type; char attributes; } KeyInfo;
typedef struct {
    uint32_t key;
    Vers vers;
    PLimit plimit;
    KeyInfo info;
    char result, status, data8;
    uint32_t data32;
    unsigned char bytes[32];
} SMCParam;

static io_connect_t conn;

static uint32_t fourcc(const char *s) { return (uint32_t)s[0] << 24 | (uint32_t)s[1] << 16 | (uint32_t)s[2] << 8 | (uint32_t)s[3]; }

static int call(SMCParam *in, SMCParam *out) {
    size_t size = sizeof(*out);
    memset(out, 0, sizeof(*out));
    kern_return_t kr = IOConnectCallStructMethod(conn, KERNEL_INDEX_SMC, in, sizeof(*in), out, &size);
    return kr == KERN_SUCCESS && out->result == 0 ? 0 : -1;
}

static int key_info(const char *key, KeyInfo *info) {
    SMCParam in = {0}, out;
    in.key = fourcc(key);
    in.data8 = CMD_READ_KEYINFO;
    if (call(&in, &out)) return -1;
    *info = out.info;
    return 0;
}

static int read_key(const char *key, KeyInfo *info, unsigned char *bytes) {
    if (key_info(key, info)) return -1;
    SMCParam in = {0}, out;
    in.key = fourcc(key);
    in.info.size = info->size;
    in.data8 = CMD_READ_BYTES;
    if (call(&in, &out)) return -1;
    memcpy(bytes, out.bytes, sizeof(out.bytes));
    return 0;
}

static int write_key(const char *key, const void *value, uint32_t size) {
    KeyInfo info;
    if (key_info(key, &info) || info.size != size) return -1;
    SMCParam in = {0}, out;
    in.key = fourcc(key);
    in.info.size = size;
    in.data8 = CMD_WRITE_BYTES;
    memcpy(in.bytes, value, size);
    return call(&in, &out);
}

static int read_float(const char *key, float *value) {
    KeyInfo info;
    unsigned char bytes[32];
    if (read_key(key, &info, bytes) || info.type != fourcc("flt ") || info.size != 4) return -1;
    memcpy(value, bytes, 4);
    return 0;
}

static int read_u8(const char *key, int *value) {
    KeyInfo info;
    unsigned char bytes[32];
    if (read_key(key, &info, bytes) || info.size != 1) return -1;
    *value = bytes[0];
    return 0;
}

static int fan_count(void) {
    int n;
    return read_u8("FNum", &n) ? 0 : n;
}

static void fan_key(char *out, int fan, const char *suffix) { snprintf(out, 5, "F%d%s", fan, suffix); }

// M5-generation SMCs spell the mode key F<n>md and have no Ftst unlock.
static void mode_key(char *out, int fan) {
    KeyInfo info;
    fan_key(out, fan, "Md");
    if (key_info(out, &info)) fan_key(out, fan, "md");
}

static int status(void) {
    int n = fan_count(), unlock = -1;
    read_u8("Ftst", &unlock);
    printf("fans %d, Ftst unlock %s\n", n, unlock < 0 ? "absent" : unlock ? "on" : "off");
    for (int i = 0; i < n; i++) {
        char k[5];
        float actual = 0, minimum = 0, maximum = 0, target = 0;
        int mode = -1;
        fan_key(k, i, "Ac"); read_float(k, &actual);
        fan_key(k, i, "Mn"); read_float(k, &minimum);
        fan_key(k, i, "Mx"); read_float(k, &maximum);
        fan_key(k, i, "Tg"); read_float(k, &target);
        mode_key(k, i); read_u8(k, &mode);
        printf("fan %d: %.0f rpm (min %.0f, max %.0f, target %.0f), %s\n", i, actual, minimum, maximum,
               target, mode == 1 ? "manual" : mode == 0 ? "auto" : "mode unknown");
    }
    return n > 0 ? 0 : 1;
}

static int set_max(void) {
    int n = fan_count(), failed = 0;
    if (!n) return fprintf(stderr, "hwgate-fan: no fans\n"), 1;
    unsigned char on = 1;
    // Apple Silicon ignores F<n>Md until the test-mode unlock is set; older SMCs have no Ftst.
    int unlockable = write_key("Ftst", &on, 1) == 0;
    for (int i = 0; i < n; i++) {
        char k[5];
        float maximum;
        fan_key(k, i, "Mx");
        if (read_float(k, &maximum)) { failed = 1; continue; }
        int ok = 0;
        // thermalmonitord takes a moment to release the fans after the unlock.
        for (int attempt = 0; attempt < 20 && !ok; attempt++) {
            mode_key(k, i);
            ok = write_key(k, &on, 1) == 0;
            fan_key(k, i, "Tg");
            ok = ok && write_key(k, &maximum, 4) == 0;
            if (!ok) usleep(250000);
        }
        failed |= !ok;
    }
    if (failed) fprintf(stderr, "hwgate-fan: could not pin every fan at max%s\n", unlockable ? "" : " (no Ftst unlock)");
    return failed;
}

static int set_auto(void) {
    int n = fan_count(), failed = 0;
    unsigned char off = 0;
    for (int i = 0; i < n; i++) {
        char k[5];
        mode_key(k, i);
        failed |= write_key(k, &off, 1) != 0;
    }
    int unlock;
    if (read_u8("Ftst", &unlock) == 0) failed |= write_key("Ftst", &off, 1) != 0;
    if (failed) fprintf(stderr, "hwgate-fan: could not return every fan to auto\n");
    return failed;
}

int main(int argc, char **argv) {
    const char *cmd = argc == 2 ? argv[1] : "";
    if (strcmp(cmd, "status") && strcmp(cmd, "max") && strcmp(cmd, "auto"))
        return fprintf(stderr, "usage: hwgate-fan status|max|auto\n"), 2;
    io_service_t smc = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"));
    if (!smc || IOServiceOpen(smc, mach_task_self(), 0, &conn) != KERN_SUCCESS)
        return fprintf(stderr, "hwgate-fan: cannot open AppleSMC\n"), 1;
    IOObjectRelease(smc);
    if (!strcmp(cmd, "status")) return status();
    if (geteuid() != 0) return fprintf(stderr, "hwgate-fan: %s needs root\n", cmd), 1;
    return !strcmp(cmd, "max") ? set_max() : set_auto();
}
