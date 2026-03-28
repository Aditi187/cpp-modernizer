#ifndef USER_H
#define USER_H

#include <stdio.h>
#include <string.h>

#define MAX_NAME 100
#define LOG_USER(u) printf("User: %s (ID: %d)\n", u.name, u.id)

typedef struct {
    int id;
    char name[MAX_NAME];
} UserRecord;

void init_user(UserRecord* u, int id, const char* name);
void print_user_time(UserRecord* u);

#endif
