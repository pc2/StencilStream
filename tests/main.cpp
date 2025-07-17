#include <catch2/catch_session.hpp>
#include <filesystem>

int main(int argc, char* argv[]) {
    // Remove any old pipe IO files.
    for (const char *path : {"0", "1", "2", "3", "4", "5", "6", "7"}) {
        if (std::filesystem::is_regular_file(path)) {
            std::remove(path);
        }
    }

    int result = Catch::Session().run(argc, argv);

    // Clean up pipe IO files.
    for (const char *path : {"0", "1", "2", "3", "4", "5", "6", "7"}) {
        if (std::filesystem::is_regular_file(path)) {
            std::remove(path);
        }
    }
    return result;
}