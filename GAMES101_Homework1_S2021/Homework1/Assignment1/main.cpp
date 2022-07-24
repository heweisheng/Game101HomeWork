#include "Triangle.hpp"
#include "rasterizer.hpp"
#include <eigen3/Eigen/Eigen>
#include <iostream>
#include <opencv2/opencv.hpp>

constexpr double MY_PI = 3.1415926;

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    //平移矩阵
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float rotation_angle)
{
    //实现思路，参考上面的函数，identity获取单位矩阵，不知道为什么要乘单位矩阵
    Eigen::Matrix4f model = Eigen::Matrix4f::Identity();
    //弧度制转换
    float rr = (float)(rotation_angle / 180.0 * MY_PI);
    Eigen::Matrix4f rotate;
    // z轴旋转矩阵，由于是绕z轴转，仅x，y变化，可以直接使用二维旋转的公式+齐次坐标
    //二维旋转矩阵推算 https://www.bilibili.com/video/BV1X7411F744?p=3&vd_source=657636f6c75d9280607e5f9ee048d9d4 18分钟
    rotate << cos(rr), -sin(rr), 0, 0,
        sin(rr), cos(rr), 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    // TODO: Implement this function
    // Create the model matrix for rotating the triangle around the Z axis.
    // Then return it.
    model = rotate * model;
    return model;
}

//作业要求时这里生成标准化矩阵
Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio,
                                      float zNear, float zFar)
{
    // Students will implement this function

    

    //标准立方体生成
    Eigen::Matrix4f projection = Eigen::Matrix4f::Identity();
    //透视投影，先把透视转成正交投影，再变回透视投影显示，利用相似三角形推算出x,y的数值
    Eigen::Matrix4f ortho;
    //公式推导 第四节 https://www.bilibili.com/video/BV1X7411F744?p=4&vd_source=657636f6c75d9280607e5f9ee048d9d4 1小时01分左右
    ortho << zNear, 0, 0, 0,
        0, zNear, 0, 0,
        0, 0, zNear + zFar, -zNear * zFar,
        0, 0, 1, 0;

    //根据三角形定理，求出left top bottom right
    float fov= eye_fov / 2;//投影平面与视野呈垂直，平面到眼睛连线，到顶部，底部的夹角为角度/2 
    float fovr =  (float)(fov * MY_PI / 180.0); //求弧度制

    

    float t = zNear * tan(fovr);//视角到顶点距离 对边/直角边=tan 对边=tan*直角边
    float b = -t;//同上 top为正方向 bottom负方向
    float r = aspect_ratio * t;//可视平面宽高比获取左右方位长度 w/h=aspect w=aspect*h r=w/2
    float l = -r;//同上 right正方向 left负方向

    Eigen::Matrix4f trans, scale,upend;
    //压缩到-1,1的矩阵
    scale << 2 / (r - l), 0, 0, 0,
        0, 2 / (t - b), 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;

    //到原点
    trans << 1, 0, 0, -(r + l) / 2,
        0, 1, 0, -(t + b) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;

    
    //内容上下颠倒 坐标系变换下
    upend<<-1,0,0,0,
    0,-1,0,0,
    0,0,1,0,
    0,0,0,1;

    
    projection = upend*trans*scale*ortho*projection;
    return projection;
    // TODO: Implement this function
    // Create the projection matrix for the given parameters.
    // Then return it.
}

int main(int argc, const char **argv)
{
    float angle = 0;
    bool command_line = false;
    std::string filename = "output.png";

    if (argc >= 3)
    {
        command_line = true;
        angle = std::stof(argv[2]); // -r by default
        if (argc == 4)
        {
            filename = std::string(argv[3]);
        }
    }

    rst::rasterizer r(700, 700);

    Eigen::Vector3f eye_pos = {0, 0, 5};

    std::vector<Eigen::Vector3f> pos{{2, 0, -2}, {0, 2, -2}, {-2, 0, -2}};

    std::vector<Eigen::Vector3i> ind{{0, 1, 2}};

    auto pos_id = r.load_positions(pos);
    auto ind_id = r.load_indices(ind);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45, 1, 0.1, 50));

        r.draw(pos_id, ind_id, rst::Primitive::Triangle);

        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::imshow("image", image);
        key = cv::waitKey(10);

        std::cout << "frame count: " << frame_count++ << '\n';

        if (key == 'a')
        {
            angle += 10;
        }
        else if (key == 'd')
        {
            angle -= 10;
        }
    }

    return 0;
}
