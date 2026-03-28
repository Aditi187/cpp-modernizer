#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <ctime>
#include <cstring>

constexpr auto MAX_SIZE = 100;

using Node = struct Node {
    int id;
    std::string name;
    std::unique_ptr<Node> next;
};

using Logger = struct Logger {
    mutable std::ofstream file;
    std::string buffer;
};

void init_logger(Logger& logger, const std::string_view filename) {
    logger.file.open(filename.data(), std::ios_base::app);
    logger.buffer = "LOG START\n";
    logger.file << logger.buffer;
}

void log_message(const Logger& logger, const std::string_view message) {
    if (logger.file.is_open()) {
        std::time_t rawtime;
        std::time(&rawtime);
#ifdef _WIN32
        std::tm timeinfo;
        localtime_s(&timeinfo, &rawtime);
#else
        std::tm* timeinfo = std::localtime(&rawtime);
#endif
        logger.file << "[" << timeinfo.tm_hour << ":" << timeinfo.tm_min << ":" << timeinfo.tm_sec << "] " << message << "\n";
        logger.file.flush();
    }
}

void close_logger(Logger& logger) {
    if (logger.file.is_open()) {
        logger.file.close();
    }
}

std::unique_ptr<Node> create_node(int id, const std::string_view name) {
    auto node = std::make_unique<Node>();
    node->id = id;
    node->name = name;
    return node;
}

void append_node(std::unique_ptr<Node>& head, int id, const std::string_view name) {
    if (!head) {
        head = create_node(id, name);
    } else {
        auto temp = head.get();
        while (temp->next) {
            temp = temp->next.get();
        }
        temp->next = create_node(id, name);
    }
}

void print_list(const std::unique_ptr<Node>& head) {
    auto temp = head.get();
    while (temp) {
        std::cout << "ID: " << temp->id << " Name: " << temp->name << "\n";
        temp = temp->next.get();
    }
}

void legacy_string_ops() {
    std::string buffer = "Legacy string example";
    int len = buffer.length();
    std::cout << "String length = " << len << "\n";
}

void legacy_array_ops() {
    std::vector<int> numbers(5);
    for (int i = 0; i < 5; i++) {
        numbers[i] = i * 10;
    }
    for (int i = 0; i < 5; i++) {
        std::cout << numbers[i] << " ";
    }
    std::cout << "\n";
}

void legacy_file_read() {
    std::ifstream file("data.txt");
    if (!file.is_open()) {
        std::cout << "file not found\n";
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::cout << line << "\n";
    }
}

class LegacyClass {
public:
    std::vector<int> values;
    LegacyClass() {
        values.resize(3);
        for (int i = 0; i < 3; i++) {
            values[i] = i * 2;
        }
    }
    void print() const {
        for (int i = 0; i < 3; i++) {
            std::cout << values[i] << "\n";
        }
    }
};

int main() {
    Logger logger;
    init_logger(logger, "app.log");
    log_message(logger, "Program started");
    std::unique_ptr<Node> head = create_node(1, "Alice");
    append_node(head, 2, "Bob");
    append_node(head, 3, "Charlie");
    print_list(head);
    legacy_string_ops();
    legacy_array_ops();
    legacy_file_read();
    LegacyClass obj;
    obj.print();
    log_message(logger, "Program finished");
    close_logger(logger);
    return 0;
}