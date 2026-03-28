#include "user.h"
#include <iostream>

int main() {
    UserRecord* user = (UserRecord*)malloc(sizeof(UserRecord));
    init_user(user, 42, "ModernUser");
    
    LOG_USER((*user));
    print_user_time(user);
    
    free(user);
    return 0;
}
