#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CLIENTS 10000
#define BUFFER_SIZE 1024

typedef void (*ClientEventCallback)(int,const char*);

struct Message
{
    int id;
    char* text;
    Message* next;
};

struct Client
{
    int id;
    char* name;
    Message* messages;
};

class EventLogger
{
private:
    FILE* fp;

public:
    EventLogger()
    {
        fp = fopen("client.log","a");
    }

    ~EventLogger()
    {
        if(fp)
        {
            fclose(fp);
        }
    }

    void log(const char* msg)
    {
        if(fp)
        {
            fprintf(fp,"%s\n",msg);
            fflush(fp);
        }
    }
};

class ClientManager
{
private:

    Client** clients;

    int count;

    int capacity;

    EventLogger* logger;

    ClientEventCallback callback;

public:

    ClientManager()
    {
        count = 0;

        capacity = 5;

        clients =
            (Client**)malloc(
                sizeof(Client*)
                * capacity);

        logger =
            new EventLogger();

        callback = NULL;
    }

    ~ClientManager()
    {
        for(int i=0;i<count;i++)
        {
            Message* msg =
                clients[i]->messages;

            while(msg)
            {
                Message* tmp = msg;

                msg = msg->next;

                free(tmp->text);

                delete tmp;
            }

            free(clients[i]->name);

            delete clients[i];
        }

        free(clients);

        delete logger;
    }

    void registerCallback(
        ClientEventCallback cb)
    {
        callback = cb;
    }

    void addClient(
        int id,
        const char* name)
    {
        if(count >= capacity)
        {
            capacity *= 2;

            clients =
                (Client**)realloc(
                    clients,
                    sizeof(Client*)
                    * capacity);
        }

        Client* c =
            new Client();

        c->id = id;

        c->name =
            (char*)malloc(
                strlen(name)+1);

        strcpy(
            c->name,
            name);

        c->messages = NULL;

        clients[count++] = c;

        logger->log(
            "Client Added");

        if(callback)
        {
            callback(
                id,
                name);
        }
    }

    void addMessage(
        int clientId,
        int msgId,
        const char* text)
    {
        for(int i=0;i<count;i++)
        {
            if(clients[i]->id
                == clientId)
            {
                Message* msg =
                    new Message();

                msg->id = msgId;

                msg->text =
                    (char*)malloc(
                        strlen(text)+1);

                strcpy(
                    msg->text,
                    text);

                msg->next =
                    clients[i]->messages;

                clients[i]->messages =
                    msg;

                logger->log(
                    "Message Added");

                return;
            }
        }
    }

    Client* findClient(
        int id)
    {
        for(int i=0;i<count;i++)
        {
            if(clients[i]->id
                == id)
            {
                return clients[i];
            }
        }

        return NULL;
    }

    void printClients()
    {
        for(int i=0;i<count;i++)
        {
            printf(
                "Client %d %s\n",
                clients[i]->id,
                clients[i]->name);

            Message* msg =
                clients[i]->messages;

            while(msg)
            {
                printf(
                    "  Msg %d: %s\n",
                    msg->id,
                    msg->text);

                msg = msg->next;
            }
        }
    }

    void save(
        const char* file)
    {
        FILE* fp =
            fopen(file,"w");

        if(!fp)
        {
            return;
        }

        for(int i=0;i<count;i++)
        {
            fprintf(
                fp,
                "CLIENT,%d,%s\n",
                clients[i]->id,
                clients[i]->name);

            Message* msg =
                clients[i]->messages;

            while(msg)
            {
                fprintf(
                    fp,
                    "MSG,%d,%d,%s\n",
                    clients[i]->id,
                    msg->id,
                    msg->text);

                msg = msg->next;
            }
        }

        fclose(fp);
    }
};

void onClientEvent(
    int id,
    const char* name)
{
    printf(
        "EVENT %d %s\n",
        id,
        name);
}

int main()
{
    ClientManager mgr;

    mgr.registerCallback(
        onClientEvent);

    mgr.addClient(
        1,
        "Alice");

    mgr.addClient(
        2,
        "Bob");

    mgr.addMessage(
        1,
        101,
        "Hello");

    mgr.addMessage(
        1,
        102,
        "World");

    mgr.addMessage(
        2,
        201,
        "Test");

    mgr.printClients();

    mgr.save(
        "clients.txt");

    return 0;
}