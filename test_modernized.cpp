#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Global pointer - a nightmare for thread safety and modernization
char* GLOBAL_LOG_BUFFER = NULL;

typedef struct {
    int id;
    char* raw_data;
    size_t len;
} LegacyRecord;
int load_record(LegacyRecord** out_rec, int id) {
    if (out_rec == nullptr || id <= 0) return -1; // Error code

    *out_rec = nullptr;
    LegacyRecord* rec = static_cast<LegacyRecord*>(malloc(sizeof(LegacyRecord)));
    if (rec == nullptr) return -1;

    rec->id = id;

    const char* dummy = "DEVICE_DATA_STREAM_0xCF";
    rec->len = strlen(dummy);
    rec->raw_data = static_cast<char*>(malloc(rec->len + 1));
    if (rec->raw_data == nullptr) {
        free(rec);
        return -1;
    }

    memcpy(rec->raw_data, dummy, rec->len + 1);
    *out_rec = rec;

    return 0; // Success code
}

void process_data() {
    LegacyRecord* rec = nullptr;
    // Pointer to pointer logic is a great test for std::expected and smart pointers
    if (load_record(&rec, 42) == 0) {
        printf("Processing Record %d: %s\n", rec->id, rec->raw_data);

        // Manual cleanup often forgotten by lazy AI
        free(rec->raw_data);
        free(rec);
    } else {
        printf("Failed to load record.\n");
    }
}

int main() {
    process_data();
    return 0;
}