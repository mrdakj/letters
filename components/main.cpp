#include "../include/image.h"
#include <fstream>
#include <thread>
#include <future>
namespace fs = std::filesystem;

std::pair<std::vector<pixel>, borders> bfs(const image& img, pixel p, std::unordered_set<pixel, pixel::hash>& visited)
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
                pixel neighbour{current_pixel.j+dj, current_pixel.i+di};
                // if neighbour is black and not yet visited
                if (img.check_color(neighbour, Color::black) && visited.find(neighbour) == visited.cend()) {
                    q.push(neighbour);
                    visited.insert(neighbour);
                }
            }
        }
    }

    return {black, b};
}

bool is_one_component(const image& img)
{
    std::unordered_set<pixel, pixel::hash> visited;

    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black)) {
                if (visited.empty()) {
                    bfs(img, p, visited);
                }
                else {
                    if (visited.find(p) == visited.cend()) {
                        return false;
                    }
                }
            }
        }
    }

    return true;
}

bool is_dot(const image& img, int weight)
{
    int threshold = (weight == 1) ? 25 : (weight == 2) ? 28 : 31;
    return (img.rows() < threshold && img.cols() < threshold);
}

void glue(std::pair<borders,image>& component, std::pair<borders,image>& dot)
{
    if (!is_one_component(component.second)) {
        return;
    }
    // std::cout << "comp " << component.first.top() << "," << component.first.left() << std::endl;
    // std::cout << "dot: " << dot.first.top() << "," << dot.first.left() << std::endl;
    // component.second.save("2.png");
    component.second.add_border(Direction::right, dot.first.right() - component.first.right());
    component.second.add_border(Direction::left, component.first.left() - dot.first.left());
    dot.second.add_border(Direction::right,  component.first.right() - dot.first.right());
    dot.second.add_border(Direction::left,  dot.first.left() - component.first.left());

    component.second.add_border(Direction::up, component.first.top() - dot.first.bottom());

    component.second = component.second.concatenate_horizontal(dot.second);

    // component.first = borders(
    //         std::min(component.first.left(), dot.first.left()),
    //         std::max(component.first.right(), dot.first.right()),
    //         dot.first.top(),
    //         component.first.bottom());
}


void bfs(const image& img, char letter, std::vector<int>& image_number, const fs::path& directory, bool prepare, int weight)
{
    std::vector<std::pair<borders,image>> components;
    std::vector<std::pair<borders,image>> dots;
    std::unordered_set<pixel, pixel::hash> visited;
    int& image_num = image_number[letter-97];

    for (int j = 0; j < img.rows(); ++j) {
        for (int i = 0; i < img.cols(); ++i) {
            pixel p{j,i};
            // if pixel is black and not yet visited
            if (img.check_color(p, Color::black) && visited.find(p) == visited.cend()) {
                auto [pixels, b] = bfs(img, p, visited);
                image im(b.height(), b.width());
                for (auto pp : pixels) {
                    im(pp.j - b.top(), pp.i - b.left()) = img(pp);
                }
                im.crop();
                if (letter == 'i' || letter == 'j') {
                    if (!is_dot(im, weight)) {
                        components.emplace_back(b, std::move(im));
                    }
                    else {
                        dots.emplace_back(b, std::move(im));
                    }
                }
                else {
                    components.emplace_back(b, std::move(im));
                }
            }
        }
    }

    if (letter == 'i' || letter == 'j') {
        for (auto& dot : dots) {
            int dot_middle = (dot.first.left() + dot.first.right()) / 2;
            auto component = std::min_element(components.begin(), components.end(), [&](auto const& lhs, auto const& rhs){
                    int lhs_result = ((lhs.first.bottom() + lhs.first.top())/2 < dot.first.top() || lhs.first.top()  > dot.first.bottom() + 80) ? 999999 : std::abs(dot_middle - (lhs.first.right()+lhs.first.left())/2);
                    int rhs_result = ((rhs.first.bottom() + rhs.first.top())/2 < dot.first.top() || rhs.first.top()  > dot.first.bottom() + 80) ? 999999 : std::abs(dot_middle - (rhs.first.right() + rhs.first.left())/2);
                    return lhs_result < rhs_result;
            });

            glue(*component, dot);
        }
    }

    for (auto& component : components) {
        auto& im = component.second;
        if ((letter == 'i' || letter == 'j') && (is_one_component(im) || im.cols() > 100)) {
            continue;
        }

        im.save(directory / std::string(1,letter) / (std::to_string(image_num) + ".png"));

        if (prepare) {
            int image_num_prepared = (image_num-1)*3 + 1;

            image im_rotated_15(im);
            image im_rotated_minus_15(im);

            im.resize(28,28);
            im.save(directory / "prepared" / std::string(1,letter) / (std::to_string(image_num_prepared) + ".png"));

            im_rotated_15.rotate(15);
            im_rotated_15.crop();
            im_rotated_15.resize(28,28);
            im_rotated_15.save(directory / "prepared" / std::string(1,letter) / (std::to_string(image_num_prepared+1) + ".png"));

            im_rotated_minus_15.rotate(-15);
            im_rotated_minus_15.crop();
            im_rotated_minus_15.resize(28,28);
            im_rotated_minus_15.save(directory / "prepared" / std::string(1,letter) / (std::to_string(image_num_prepared+2) + ".png"));
        }

        ++image_num;
    }
}

void get_images(const fs::path& input_dir, bool prepare, int weight)
{
    std::vector<int> image_number(26,1);

    fs::path output_dir = input_dir;
    for(auto& p: fs::directory_iterator(input_dir)) {
        if (p.path().extension() == ".png") {
            char letter = p.path().filename().string()[0];
            fs::create_directories(output_dir / std::string(1, letter));
            if (prepare) {
                fs::create_directories(input_dir / "prepared" / std::string(1, letter));
            }
            auto image_name = p.path().string();
            std::cout << image_name << std::endl;
            image img(image_name);
            img.threshold();
            img.crop();
            bfs(img, letter, image_number, output_dir, prepare, weight);
        }
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "./main path_to_dir [prepare]" << std::endl;
        return -1;
    }

    bool prepare = false;
    fs::path input_dir(argv[1]);

    if (argc == 3) {
        prepare = std::string(argv[2]) == std::string("prepare");
    }

    std::vector<std::future<void>> threads(3);
    threads[0] = std::async([&]{ std::invoke(get_images, input_dir / "normal", prepare, 1); });
    threads[1] = std::async([&]{ std::invoke(get_images, input_dir / "medium", prepare, 2); });
    threads[2] = std::async([&]{ std::invoke(get_images, input_dir / "bold", prepare, 3); });

    return 0;
}
