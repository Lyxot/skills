// hwgate-fan: pin every fan at its maximum for a --stable run, or hand them back to macOS.
//
//   hwgate-fan status   read-only; works as any user
//   hwgate-fan max      root: unlock manual mode (Ftst) and target each fan's F<n>Mx
//   hwgate-fan auto     root: return each fan to automatic and drop the unlock
//
// Only ever raises fan speed: there is no command that sets a target below the maximum.
#include <IOKit/IOKitLib.h>
#include <fcntl.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <sys/file.h>
#include <sys/stat.h>
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

// Apple Silicon ignores F<n>Md until the test-mode unlock is set; M5-era SMCs have no Ftst.
static void request(int n, int manual) {
    unsigned char flag = manual;
    float maximum[n];
    char k[5];
    // No fan leaves automatic unless every maximum is known: manual at an old target is worse than auto.
    for (int i = 0; i < n && manual; i++) {
        fan_key(k, i, "Mx");
        if (read_float(k, &maximum[i])) return;
    }
    if (manual) write_key("Ftst", &flag, 1);
    for (int i = 0; i < n; i++) {
        fan_key(k, i, "Tg");
        if (manual && write_key(k, &maximum[i], 4)) continue;
        mode_key(k, i);
        write_key(k, &flag, 1);
    }
    int unlock;
    if (!manual && read_u8("Ftst", &unlock) == 0) write_key("Ftst", &flag, 1);
}

static int settled(int n, int manual) {
    int unlock, mode;
    if (read_u8("Ftst", &unlock) == 0 && unlock != manual) return 0;
    for (int i = 0; i < n; i++) {
        char k[5];
        float maximum, target;
        mode_key(k, i);
        if (read_u8(k, &mode) || mode != manual) return 0;
        fan_key(k, i, "Mx");
        if (manual && read_float(k, &maximum)) return 0;
        fan_key(k, i, "Tg");
        if (manual && (read_float(k, &target) || target < maximum - 1)) return 0;
    }
    return 1;
}

// The SMC applies writes up to ~0.8s after accepting them, so return only once the state has held
// for longer than that: otherwise writes still pending from an earlier, possibly killed, call can
// land after this one returns and undo it.
static int set_mode(int manual) {
    int n = fan_count(), held = 0;
    if (!n) return fprintf(stderr, "hwgate-fan: no fans\n"), 1;
    for (int tick = 0; tick < 100; tick++) {
        if (!held && tick % 10 == 0) request(n, manual);  // re-issued each second until it holds
        usleep(100000);
        held = settled(n, manual) ? held + 1 : 0;
        if (held >= 15) return 0;
    }
    fprintf(stderr, "hwgate-fan: fans did not hold %s for 1.5s within 10s\n", manual ? "max" : "auto");
    // A partial pin is handed back rather than left for the whole run.
    if (manual) set_mode(0);
    return 1;
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
    // One max/auto at a time: killing sudo does not reach the helper (SIGKILL is not relayed), so an
    // orphaned max would otherwise keep fighting the auto after it. The lock file is root-only (0600, in
    // the directory install-fan-helper.sh verifies), so no other account can hold it.
    char path[4096];
    uint32_t size = sizeof(path) - 5;
    if (_NSGetExecutablePath(path, &size)) return fprintf(stderr, "hwgate-fan: path too long\n"), 1;
    strcat(path, ".lock");
    int lock = open(path, O_RDWR | O_CREAT | O_NOFOLLOW, 0600);
    if (lock < 0 || fchown(lock, 0, 0) || fchmod(lock, 0600) || flock(lock, LOCK_EX)) return fprintf(stderr, "hwgate-fan: cannot lock %s\n", path), 1;
    int manual = !strcmp(cmd, "max");
    // A max whose caller is gone was abandoned, and the auto that followed may already have run.
    if (manual && getppid() == 1) return fprintf(stderr, "hwgate-fan: max abandoned by its caller\n"), 1;
    return set_mode(manual);
}
