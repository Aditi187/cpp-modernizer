#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <mutex>

constexpr auto MAX_USERS = 10;
constexpr auto BUFFER_SIZE = 256;

inline constexpr int multiply(int a, int b) {
    return a * b;
}

struct User {
    int id;
    std::string name;
    std::unique_ptr<User> next;
};

struct Logger {
    std::ofstream file;
    mutable std::mutex mutex;
    std::string buffer;
};

void init_logger(Logger& logger, const std::string& filename) {
    logger.file.open(filename, std::ios_base::app);
    logger.buffer = "LOGGER INITIALIZED\n";
    logger.file << logger.buffer;
}

void log_message(Logger& logger, const std::string& msg) {
    if (!logger.file.is_open()) return;
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    auto now_tm = *std::localtime(&now_time);
    std::lock_guard<std::mutex> lock(logger.mutex);
    logger.file << "[" << now_tm.tm_hour << ":" << now_tm.tm_min << ":" << now_tm.tm_sec << "] " << msg << "\n";
    logger.file.flush();
}

void close_logger(Logger& logger) {
    logger.file.close();
}

std::unique_ptr<User> create_user(int id, const std::string& name) {
    auto u = std::make_unique<User>();
    u->id = id;
    u->name = name;
    return u;
}

void append_user(User& head, int id, const std::string& name) {
    auto temp = &head;
    while (temp->next) {
        temp = temp->next.get();
    }
    temp->next = create_user(id, name);
}

void print_users(const User& head) {
    const auto* temp = &head;
    while (temp) {
        std::cout << "USER " << temp->id << " : " << temp->name << "\n";
        temp = temp->next.get();
    }
}

void legacy_string_ops() {
    std::string buffer = "legacy_string";
    buffer += "_suffix";
    std::cout << "buffer=" << buffer << " length=" << buffer.length() << "\n";
}

std::vector<int> create_array(int n) {
    std::vector<int> arr(n);
    for (int i = 0; i < n; i++) {
        arr[i] = multiply(i, 2);
    }
    return arr;
}

void print_array(const std::vector<int>& arr) {
    for (const auto& value : arr) {
        std::cout << value << " ";
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

class LegacyBuffer {
public:
    LegacyBuffer(int s) : data("class_buffer") {}
    void print() const {
        std::cout << "buffer=" << data << "\n";
    }
private:
    std::string data;
};

class LegacyIntArray {
public:
    LegacyIntArray(int n) : values(n) {
        for (int i = 0; i < n; i++) {
            values[i] = i * 3;
        }
    }
    void show() const {
        for (const auto& value : values) {
            std::cout << value << "\n";
        }
    }
private:
    std::vector<int> values;
};

void manual_copy(std::string& dst, const std::string& src) {
    dst = src;
}

void unsafe_concat() {
    std::string a = "unsafe";
    a += "concat";
    std::cout << a << "\n";
}

void mixed_operations() {
    auto numbers = create_array(5);
    print_array(numbers);
}

int main() {
    Logger logger;
    init_logger(logger, "app.log");
    log_message(logger, "PROGRAM START");
    auto head = create_user(1, "Alice");
    append_user(*head, 2, "Bob");
    append_user(*head, 3, "Charlie");
    print_users(*head);
    legacy_string_ops();
    legacy_file_read();
    mixed_operations();
    LegacyBuffer buffer(50);
    buffer.print();
    LegacyIntArray arr(5);
    arr.show();
    unsafe_concat();
    std::string temp;
    manual_copy(temp, "manual_copy_test");
    std::cout << temp << "\n";
    log_message(logger, "PROGRAM END");
    close_logger(logger);
    return 0;
}