#include "../include/image.h"
#include <fstream>
#include <thread>
#include <future>
namespace fs = std::filesystem;

void get_images_train(char letter)
{
    int counter = 1;

    fs::path output_dir = "../../dataset/train/one";
    fs::create_directories(output_dir / std::string(1,letter));

    auto create = [&](const fs::path& dir)
    {
        for(auto& p: fs::directory_iterator(dir / std::string(1,letter))) {
            image img(p.path());
            img.resize(28,28);
            img.save(output_dir / std::string(1,letter) / (std::to_string(counter) + ".png"));
            ++counter;
        }
    };

    create("../../dataset/train/one_letter/normal");
    create("../../dataset/train/one_letter/normal_rotated");
    create("../../dataset/train/one_letter/medium");
    create("../../dataset/train/one_letter/medium_rotated");
    create("../../dataset/train/one_letter/bold");
    create("../../dataset/train/one_letter/bold_rotated");

    std::cout << "done " << std::string(1,letter) << std::endl;
}

void get_images_validation(char letter)
{
    int counter = 1;

    fs::path output_dir = "../../dataset/validation/one";
    fs::create_directories(output_dir / std::string(1,letter));

    auto create = [&](const fs::path& dir)
    {
        for(auto& p: fs::directory_iterator(dir / std::string(1,letter))) {
            image img(p.path());
            img.resize(28,28);
            img.save(output_dir / std::string(1,letter) / (std::to_string(counter) + ".png"));
            ++counter;
        }
    };

    create("../../dataset/validation/one_letter/normal");
    create("../../dataset/validation/one_letter/normal_rotated");
    create("../../dataset/validation/one_letter/medium");
    create("../../dataset/validation/one_letter/medium_rotated");
    create("../../dataset/validation/one_letter/bold");
    create("../../dataset/validation/one_letter/bold_rotated");

    std::cout << "done " << std::string(1,letter) << std::endl;
}

void get_images_train1()
{
    get_images_train('a');
    get_images_train('b');
    get_images_train('c');
    get_images_train('d');
    get_images_train('e');
    get_images_train('f');
}

void get_images_train2()
{
    get_images_train('g');
    get_images_train('h');
    get_images_train('i');
    get_images_train('j');
    get_images_train('k');
    get_images_train('l');
    get_images_train('m');
}

void get_images_train3()
{
    get_images_train('n');
    get_images_train('o');
    get_images_train('p');
    get_images_train('q');
    get_images_train('r');
    get_images_train('s');
}

void get_images_train4()
{
    get_images_train('t');
    get_images_train('u');
    get_images_train('v');
    get_images_train('w');
    get_images_train('x');
    get_images_train('y');
    get_images_train('z');
}

void get_images_validation1()
{
    get_images_validation('a');
    get_images_validation('b');
    get_images_validation('c');
    get_images_validation('d');
    get_images_validation('e');
    get_images_validation('f');
}

void get_images_validation2()
{
    get_images_validation('g');
    get_images_validation('h');
    get_images_validation('i');
    get_images_validation('j');
    get_images_validation('k');
    get_images_validation('l');
    get_images_validation('m');
}

void get_images_validation3()
{
    get_images_validation('n');
    get_images_validation('o');
    get_images_validation('p');
    get_images_validation('q');
    get_images_validation('r');
    get_images_validation('s');
}

void get_images_validation4()
{
    get_images_validation('t');
    get_images_validation('u');
    get_images_validation('v');
    get_images_validation('w');
    get_images_validation('x');
    get_images_validation('y');
    get_images_validation('z');
}


int main(int argc, char** argv)
{
    bool train = true;
    if (argc == 2) {
        train = std::string(argv[1]) == std::string("train");
    }

    if (train) {
        fs::create_directories("../../dataset/train/one");
    }
    else {
        fs::create_directories("../../dataset/validation/one");
    }

    std::vector<std::future<void>> threads(4);
    if (train) {
        threads[0] = std::async([&]{ std::invoke(get_images_train1); });
        threads[1] = std::async([&]{ std::invoke(get_images_train2); });
        threads[2] = std::async([&]{ std::invoke(get_images_train3); });
        threads[3] = std::async([&]{ std::invoke(get_images_train4); });
    }
    else {
        threads[0] = std::async([&]{ std::invoke(get_images_validation1); });
        threads[1] = std::async([&]{ std::invoke(get_images_validation2); });
        threads[2] = std::async([&]{ std::invoke(get_images_validation3); });
        threads[3] = std::async([&]{ std::invoke(get_images_validation4); });
    }

    return 0;
}
