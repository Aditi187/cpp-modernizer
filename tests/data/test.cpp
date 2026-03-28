#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_USERS 10
#define BUFFER_SIZE 256
#define MULTIPLY(a,b) ((a)*(b))
#define SAFE_FREE(p) if(p){free(p);p=NULL;}

typedef struct User {
    int id;
    char name[50];
    struct User* next;
} User;

typedef struct Logger {
    FILE* file;
    char* buffer;
} Logger;

void init_logger(Logger* logger,const char* filename){
    logger->file=fopen(filename,"a");
    logger->buffer=(char*)malloc(BUFFER_SIZE);

    if(logger->buffer!=NULL){
        strcpy(logger->buffer,"LOGGER INITIALIZED\n");
        fprintf(logger->file,"%s",logger->buffer);
    }
}

void log_message(Logger* logger,const char* msg){

    if(logger->file==NULL) return;

    time_t rawtime;
    struct tm* timeinfo;

    time(&rawtime);
    timeinfo=localtime(&rawtime);

    fprintf(
        logger->file,
        "[%d:%d:%d] %s\n",
        timeinfo->tm_hour,
        timeinfo->tm_min,
        timeinfo->tm_sec,
        msg
    );

    fflush(logger->file);
}

void close_logger(Logger* logger){

    if(logger->file){
        fclose(logger->file);
    }

    SAFE_FREE(logger->buffer);
}


User* create_user(int id,const char* name){

    User* u=(User*)malloc(sizeof(User));

    u->id=id;

    strcpy(u->name,name);

    u->next=NULL;

    return u;
}

void append_user(User* head,int id,const char* name){

    User* temp=head;

    while(temp->next!=NULL){

        temp=temp->next;
    }

    temp->next=create_user(id,name);
}

void print_users(User* head){

    User* temp=head;

    while(temp!=NULL){

        printf("USER %d : %s\n",temp->id,temp->name);

        temp=temp->next;
    }
}

void free_users(User* head){

    User* temp=head;

    while(temp!=NULL){

        User* next=temp->next;

        free(temp);

        temp=next;
    }
}


void legacy_string_ops(){

    char buffer[BUFFER_SIZE];

    strcpy(buffer,"legacy_string");

    strcat(buffer,"_suffix");

    printf("buffer=%s length=%d\n",buffer,strlen(buffer));
}


int* create_array(int n){

    int* arr=(int*)malloc(sizeof(int)*n);

    for(int i=0;i<n;i++){

        arr[i]=MULTIPLY(i,2);
    }

    return arr;
}

void print_array(int* arr,int n){

    for(int i=0;i<n;i++){

        printf("%d ",arr[i]);
    }

    printf("\n");
}


void legacy_file_read(){

    FILE* file=fopen("data.txt","r");

    if(file==NULL){

        printf("file not found\n");

        return;
    }

    char line[128];

    while(fgets(line,sizeof(line),file)){

        printf("%s",line);
    }

    fclose(file);
}


class LegacyBuffer {

private:

    char* data;

    int size;

public:

    LegacyBuffer(int s){

        size=s;

        data=new char[size];

        strcpy(data,"class_buffer");

    }

    ~LegacyBuffer(){

        delete[] data;
    }

    void print(){

        printf("buffer=%s\n",data);
    }
};


class LegacyIntArray {

private:

    int* values;

    int length;

public:

    LegacyIntArray(int n){

        length=n;

        values=new int[length];

        for(int i=0;i<length;i++){

            values[i]=i*3;
        }
    }

    ~LegacyIntArray(){

        delete[] values;
    }

    void show(){

        for(int i=0;i<length;i++){

            printf("%d\n",values[i]);
        }
    }
};


void manual_copy(char* dst,const char* src){

    while(*src){

        *dst=*src;

        dst++;

        src++;
    }

    *dst='\0';
}


void unsafe_concat(){

    char a[50]="unsafe";

    char b[50]="concat";

    strcat(a,b);

    printf("%s\n",a);
}


void mixed_operations(){

    int* numbers=create_array(5);

    print_array(numbers,5);

    free(numbers);
}


int main(){

    Logger logger;

    init_logger(&logger,"app.log");

    log_message(&logger,"PROGRAM START");



    User* head=create_user(1,"Alice");

    append_user(head,2,"Bob");

    append_user(head,3,"Charlie");

    print_users(head);



    legacy_string_ops();

    legacy_file_read();

    mixed_operations();



    LegacyBuffer buffer(50);

    buffer.print();



    LegacyIntArray arr(5);

    arr.show();



    unsafe_concat();



    char temp[50];

    manual_copy(temp,"manual_copy_test");

    printf("%s\n",temp);



    log_message(&logger,"PROGRAM END");



    close_logger(&logger);

    free_users(head);



    return 0;
}