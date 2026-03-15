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

// Hard to modernize: returns raw pointer, uses malloc, and return codes for errors
int load_record(LegacyRecord** out_rec, int id) {
    if (id <= 0) return -1; // Error code

    *out_rec = (LegacyRecord*)malloc(sizeof(LegacyRecord));
    (*out_rec)->id = id;
    
    const char* dummy = "DEVICE_DATA_STREAM_0xCF";
    (*out_rec)->len = strlen(dummy);
    (*out_rec)->raw_data = (char*)malloc((*out_rec)->len + 1);
    strcpy((*out_rec)->raw_data, dummy);
    
    return 0; // Success code
}

void process_data() {
    LegacyRecord* rec = NULL;
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