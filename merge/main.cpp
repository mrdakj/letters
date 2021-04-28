#include "../include/image.h"
#include <algorithm>
#include <optional>
#include <fstream>
#include <unordered_map>
#include <thread>
#include <future>
#include <random>
namespace fs = std::filesystem;

enum class weight { normal, medium, bold }; 

fs::path get_dir(weight w)
{
    return  (w == weight::normal) ? "../../dataset/train/one_letter/normal" :
            (w == weight::medium) ? "../../dataset/train/one_letter/medium" :
                                    "../../dataset/train/one_letter/bold";
}

int letter_index(char letter)
{
    return letter - 97;
}

std::string str(char x)
{
    return std::string(1,x);
}


int random(int l, int r)
{
    // returns random from [l,r]
    return rand()%(r-l+1) + l;
}

bool is_long_up(char label)
{
    std::vector<char> long_letters = {'b', 'd', 'l', 'f', 't', 'h', 'k'};
    return (std::any_of(long_letters.begin(), long_letters.end(), [&](char x) { return x == label; }));
}

bool is_long_down(char label)
{
    std::vector<char> long_letters = {'g', 'y', 'q', 'j', 'p'};
    return (std::any_of(long_letters.begin(), long_letters.end(), [&](char x) { return x == label; }));
}

bool is_long_letter(char label)
{
    return is_long_up(label) || is_long_down(label);
}

bool check_pair(const image& img_a, const image& img_b, char label_a, char label_b)
{
    if (img_a.rows() < 10 || img_a.cols() < 10 || img_b.rows() < 10 || img_b.cols() < 10) {
        return false;
    }

    if (!is_long_letter(label_a) && !is_long_letter(label_b)) {
        return std::abs(img_a.rows() - img_b.rows()) < std::min(img_a.rows(), img_b.rows())/3;
    }

    if (is_long_down(label_a) && is_long_down(label_b)) {
        return std::abs(img_a.rows() - img_b.rows()) < std::min(img_a.rows(), img_b.rows())/3;
    }

    if (is_long_up(label_a) && is_long_up(label_b)) {
        return std::abs(img_a.rows() - img_b.rows()) < std::min(img_a.rows(), img_b.rows())/3;
    }

    if (is_long_up(label_a) && !is_long_letter(label_b)) {
        return img_a.rows() > img_b.rows();
    }

    if (is_long_down(label_a) && !is_long_letter(label_b)) {
        return img_a.rows() > img_b.rows();
    }

    if (is_long_up(label_b) && !is_long_letter(label_a)) {
        return img_b.rows() > img_a.rows();
    }

    if (is_long_down(label_b) && !is_long_letter(label_a)) {
        return img_b.rows() > img_a.rows();
    }

    return true;
}

void process_image(image& img)
{
    img.threshold(100);
    img.crop();
}

void bfs(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited)
{
    std::queue<pixel> q;
    q.push(p);
    visited.insert(p);

    std::vector<pixel> black;

    // left, right, top, bottm
    borders b{p.i, p.i, p.j, p.j};

    while (!q.empty()) {
        auto current_pixel = q.front();
        black.push_back(current_pixel);
        q.pop();
        b.update(current_pixel);

        for (int dj : {-1, 0, 1}) {
            for (int di : {-1, 0, 1}) {
                if (std::abs(dj) == std::abs(di)) {
                    continue;
                }
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                // if neighbour is black and not yet visited
                if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }
}

bool number_of_components(const image& img, int expected_number_of_components)
{
    int result = 0;
    std::unordered_set<pixel, pixel::hash> visited;

    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black)) {
                if (visited.find(p) == visited.cend()) {
                    ++result;
                    if (result > expected_number_of_components) {
                        return false;
                    }
                    bfs(img, p, visited);
                }
            }
        }
    }

    return result == expected_number_of_components;
}

std::pair<int,int> get_offsets(const image& img_a, const image& img_b, char label_a, char label_b)
{
    int offset_a = 0;
    int offset_b = 0;

    if (!is_long_down(label_a) && !is_long_down(label_b))
    {
        int choose = random(0,1);
        if (choose == 0) {
            offset_a = random(0, 10);
        }
        else {
            offset_b = random(0, 10);
        }
    }

    if (is_long_down(label_a) && is_long_down(label_b))
    {
        // jj
        int choose = random(0,1);
        if (choose == 0) {
            offset_a = random(0,10);
        }
        else {
            offset_b = random(0,10);
        }
    }

    if (is_long_down(label_a) && !is_long_letter(label_b))
    {
        // ja
        int choose = random(0,1);
        if (choose == 0) {
            offset_a = random(0,10);
        }
        else {
            offset_b = random(0,10);
        }
    }

    if (!is_long_letter(label_a) && is_long_down(label_b))
    {
        // aj
        int choose = random(0,1);
        if (choose == 0) {
            offset_a = random(0,10);
        }
        else {
            offset_b = random(0,10);
        }
    }

    if (is_long_down(label_a) && is_long_up(label_b))
    {
        // jb
        offset_a = img_b.rows()/3 + random(-2,5);
    }

    if (is_long_up(label_a) && is_long_down(label_b))
    {
        // bj
        offset_b = img_a.rows()/3 + random(-2,5);
    }

    return {offset_a, offset_b};
}

image merge(const image& img_a, const image& img_b, char label_a, char label_b, int step, int offset_a, int offset_b)
{
    image result(img_a.rows()+img_b.rows()+offset_a+offset_b, img_a.cols()+img_b.cols());

    int width_a = img_a.cols();
    int height_result = result.rows();

    if (img_a.rows() + offset_a >= height_result) {
        offset_a = 0;
    }
    if (img_b.rows() + offset_b >= height_result) {
        offset_b = 0;
    }
            
    if (!is_long_down(label_a) && !is_long_down(label_b)) {
        int current_result_j = height_result-1;
        for (int j = img_a.rows()-1; j >= 0; --j) {
            for (int i = 0; i < img_a.cols(); ++i) {
                result(current_result_j-offset_a,i) = img_a(j,i);
            }
            --current_result_j;
        }

        current_result_j = height_result-1;
        for (int j = img_b.rows()-1; j >= 0 ; --j) {
            for (int i = 0; i < img_b.cols(); ++i) {
                pixel p{current_result_j-offset_b,i+width_a-step};
                if (result.check_color(p, Color::white)) {
                    result(p) = img_b(j,i);
                }
            }
            --current_result_j;
        }
    }
    else {
        for (int j = 0; j < img_a.rows(); ++j) {
            for (int i = 0; i < img_a.cols(); ++i) {
                result(j+offset_a,i) = img_a(j,i);
            }
        }

        for (int j = 0; j < img_b.rows(); ++j) {
            for (int i = 0; i < img_b.cols(); ++i) {
                pixel p{j+offset_b,i+width_a-step};
                if (result.check_color(p, Color::white)) {
                    result(p) = img_b(j,i);
                }
            }
        }
    }

    result.crop();
    return result;
}

std::optional<image> merge(const image& img_a, const image& img_b, char label_a, char label_b)
{
    int expected_number_of_components = 0;
    if (label_a != 'i' && label_a != 'j') {
        if (label_b != 'i' && label_b != 'j') {
            expected_number_of_components = 1;
        }
        else {
            expected_number_of_components = 2;
        }
    }
    else {
        if (label_b != 'i' && label_b != 'j') {
            expected_number_of_components = 2;
        }
        else {
            expected_number_of_components = 3;
        }
    }

    int step = 0;
    auto [offset_a, offset_b] = get_offsets(img_a, img_b, label_a, label_b);
    while (true) {
        auto result = merge(img_a, img_b, label_a, label_b, step, offset_a, offset_b);
        step += 1;
        if (step>=img_a.rows() || result.cols() - step < 0) {
            return std::nullopt;
        }

        if (number_of_components(result, expected_number_of_components)) {
            int random_step = random(0,3);
            if (random_step)
            {
                result = merge(img_a, img_b, label_a, label_b, step+random_step, offset_a, offset_b);
            }
            return std::optional<image>(result);
        }
    }

    return std::nullopt;
}

bool empty(const image& img)
{
    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            if (img.check_color({j,i}, Color::black)) {
                return false;
            }
        }
    }

    return true;
}

void prepare(image& img)
{
    img.resize(28,28);
}

std::size_t number_of_files_in_directory(const fs::path& path)
{
    using fs::directory_iterator;
    return std::distance(directory_iterator(path), directory_iterator{});
}

std::vector<int> number_of_files(const fs::path& path)
{
    std::vector<int> result(26,1);
    std::vector<char> letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    for (char letter : letters) {
        result[letter-97] = number_of_files_in_directory(path / std::string(1,letter));
    }

    return result;
}

void create_data(char a, const fs::path& dir, const fs::path& out_dir, int& current_count)
{
    auto letter_count = number_of_files(dir);
    std::vector<char> letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};

    auto generate_pair = [&](char a, char b)
    {
        fs::path direcotry_a = dir;
        fs::path direcotry_b = dir;
        // int rotated_a = random(0,1);
        // if (rotated_a) {
        //     direcotry_a = fs::path(direcotry_a.string() + "_rotated");
        // }
        // int rotated_b = random(0,1);
        // if (rotated_b) {
        //     direcotry_b = fs::path(direcotry_b.string() + "_rotated");
        // }
        // int limit_a = (rotated_a) ? 2*letter_count[a-97] : letter_count[a-97];
        // int limit_b = (rotated_b) ? 2*letter_count[b-97] : letter_count[b-97];
        int limit_a =  letter_count[a-97];
        int limit_b =  letter_count[b-97];

        std::string path_a = direcotry_a / std::string(1,a) / (std::to_string(random(1,limit_a)) + ".png");
        std::string path_b = direcotry_b / std::string(1,b) / (std::to_string(random(1,limit_b)) + ".png");
        image img_a(path_a);
        image img_b(path_b);
        while (!check_pair(img_a, img_b, a, b)) {
            path_a = direcotry_a / std::string(1,a) / (std::to_string(random(1,limit_a)) + ".png");
            path_b = direcotry_b / std::string(1,b) / (std::to_string(random(1,limit_b)) + ".png");
            img_a = image(path_a);
            img_b = image(path_b);
        }
        process_image(img_a);
        process_image(img_b);
        return merge(img_a, img_b, a, b);
    };

    for (char b : letters) {
        for (int i = 0; i < 200; i++) {
            std::optional<image> merged = std::nullopt;
            do {
                merged = generate_pair(a, b);
                if (merged) {
                    prepare(*merged);
                    (*merged).save(out_dir / str(a) / (std::to_string(current_count) + ".png"));
                    ++current_count;
                }
            } while(!merged);
        }
    }
}

void create_data(char a)
{
    fs::path out_dir("../../dataset/train/two_letters_combined/first");
    int current_count = 1;
    create_data(a, get_dir(weight::normal), out_dir, current_count);
    create_data(a, get_dir(weight::medium), out_dir, current_count);
    create_data(a, get_dir(weight::bold), out_dir, current_count);
    std::cout << "done " << str(a) << std::endl;
}


void create1()
{
    create_data('a');
    create_data('b');
    create_data('c');
    create_data('d');
    create_data('e');
    create_data('f');
}

void create2()
{
    create_data('g');
    create_data('h');
    create_data('i');
    create_data('j');
    create_data('k');
    create_data('l');
    create_data('m');
}

void create3()
{
    create_data('n');
    create_data('o');
    create_data('p');
    create_data('q');
    create_data('r');
    create_data('s');
}

void create4()
{
    create_data('t');
    create_data('u');
    create_data('v');
    create_data('w');
    create_data('x');
    create_data('y');
    create_data('z');
}

int main(int argc, char** argv)
{
    fs::path out_dir("../../dataset/train/two_letters_combined/first");
    fs::create_directories(out_dir);

    std::vector<char> letters = {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'};
    for (char letter : letters) {
        fs::create_directories(out_dir / str(letter));
    }

    std::vector<std::future<void>> threads(4);
    threads[0] = std::async([&]{ std::invoke(create1); });
    threads[1] = std::async([&]{ std::invoke(create2); });
    threads[2] = std::async([&]{ std::invoke(create3); });
    threads[3] = std::async([&]{ std::invoke(create4); });

    return 0;
}
