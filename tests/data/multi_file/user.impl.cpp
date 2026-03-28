#include "user.h"
#include <stdlib.h>
#include <time.h>

void init_user(UserRecord* u, int id, const char* name) {
    u->id = id;
    if (name) {
        strncpy(u->name, name, MAX_NAME - 1);
        u->name[MAX_NAME - 1] = '\0';
    }
}

void print_user_time(UserRecord* u) {
    time_t now = time(NULL);
    struct tm* tm_info = localtime(&now);
    printf("User %s accessed at: %02d:%02d:%02d\n", 
           u->name, tm_info->tm_hour, tm_info->tm_min, tm_info->tm_sec);
}
