#include <iostream>
#include <opencv2/opencv.hpp>

#include "global.hpp"
#include "rasterizer.hpp"
#include "Triangle.hpp"
#include "Shader.hpp"
#include "Texture.hpp"
#include "OBJ_Loader.h"

Eigen::Matrix4f get_view_matrix(Eigen::Vector3f eye_pos)
{
    Eigen::Matrix4f view = Eigen::Matrix4f::Identity();

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, -eye_pos[0],
        0, 1, 0, -eye_pos[1],
        0, 0, 1, -eye_pos[2],
        0, 0, 0, 1;

    view = translate * view;

    return view;
}

Eigen::Matrix4f get_model_matrix(float angle)
{
    Eigen::Matrix4f rotation;
    angle = angle * MY_PI / 180.f;
    rotation << cos(angle), 0, sin(angle), 0,
        0, 1, 0, 0,
        -sin(angle), 0, cos(angle), 0,
        0, 0, 0, 1;

    Eigen::Matrix4f scale;
    scale << 2.5, 0, 0, 0,
        0, 2.5, 0, 0,
        0, 0, 2.5, 0,
        0, 0, 0, 1;

    Eigen::Matrix4f translate;
    translate << 1, 0, 0, 0,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    return translate * rotation * scale;
}

Eigen::Matrix4f get_projection_matrix(float eye_fov, float aspect_ratio, float zNear, float zFar)
{
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
    float fov = eye_fov / 2;                   //投影平面与视野呈垂直，平面到眼睛连线，到顶部，底部的夹角为角度/2
    float fovr = (float)(fov * MY_PI / 180.0); //求弧度制

    float t = zNear * tan(fovr); //视角到顶点距离 对边/直角边=tan 对边=tan*直角边
    float b = -t;                //同上 top为正方向 bottom负方向
    float r = aspect_ratio * t;  //可视平面宽高比获取左右方位长度 w/h=aspect w=aspect*h r=w/2
    float l = -r;                //同上 right正方向 left负方向

    Eigen::Matrix4f trans, scale, upend;
    //到原点
    trans << 1, 0, 0, -(r + l) / 2,
        0, 1, 0, -(t + b) / 2,
        0, 0, 1, -(zNear + zFar) / 2,
        0, 0, 0, 1;

    //压缩到-1,1的矩阵
    scale << 2 / (r - l), 0, 0, 0,
        0, 2 / (t - b), 0, 0,
        0, 0, 2 / (zNear - zFar), 0,
        0, 0, 0, 1;

    //内容上下颠倒 坐标系变换下
    upend << -1, 0, 0, 0,
        0, -1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1;

    projection = upend * scale * trans * ortho * projection;
    return projection;
}

Eigen::Vector3f vertex_shader(const vertex_shader_payload &payload)
{
    return payload.position;
}

Eigen::Vector3f normal_fragment_shader(const fragment_shader_payload &payload)
{
    Eigen::Vector3f return_color = (payload.normal.head<3>().normalized() + Eigen::Vector3f(1.0f, 1.0f, 1.0f)) / 2.f;
    Eigen::Vector3f result;
    result << return_color.x() * 255, return_color.y() * 255, return_color.z() * 255;
    return result;
}

static Eigen::Vector3f reflect(const Eigen::Vector3f &vec, const Eigen::Vector3f &axis)
{
    auto costheta = vec.dot(axis);
    return (2 * costheta * axis - vec).normalized();
}

struct light
{
    Eigen::Vector3f position;
    Eigen::Vector3f intensity;
};

Eigen::Vector3f texture_fragment_shader(const fragment_shader_payload &payload)
{
    Eigen::Vector3f return_color = {0, 0, 0};
    if (payload.texture)
    {
        // TODO: Get the texture value at the texture coordinates of the current fragment
        return_color = payload.texture->getColor(payload.tex_coords.x(), payload.tex_coords.y());
    }
    Eigen::Vector3f texture_color;
    texture_color << return_color.x(), return_color.y(), return_color.z();

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = texture_color / 255.f;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = texture_color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights)
    {
        Eigen::Vector3f vlight = light.position - point ;
        //漫反射 kd*(I/r^2)*max(0,normal(单位向量)点乘光线向量(单位向量))
        float cosnum = std::max(0.0f, normal.dot(vlight.normalized()));
        if (cosnum > 0.0f)
        {
            result_color+=kd.cwiseProduct((light.intensity / vlight.squaredNorm())*cosnum);
        }
        Eigen::Vector3f veye= eye_pos - point;
        //高光 ka*(I/r^2)*max(0,pow(normal(单位向量)点乘半程向量(单位向量),p))
        float halfcross=std::pow(std::max(0.0f, normal.dot((veye+vlight).normalized())),p);
        if(halfcross > 0.0f)
        {
            result_color+=ks.cwiseProduct((light.intensity / vlight.squaredNorm())*halfcross);
        }
        //环境光 Ka*Ia
        result_color+=ka.cwiseProduct(amb_light_intensity);
    }

    return result_color * 255.f;
}

Eigen::Vector3f phong_fragment_shader(const fragment_shader_payload &payload)
{
    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);    //环境光
    Eigen::Vector3f kd = payload.color;                           //漫反射
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937); //高光

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    Eigen::Vector3f result_color = {0, 0, 0};
    for (auto &light : lights)
    {
        Eigen::Vector3f vlight = light.position - point;
        //漫反射 kd*(I/r^2)*max(0,normal(单位向量)点乘光线向量(单位向量))
        float cosnum = std::max(0.0f, normal.dot(vlight.normalized()));
        if (cosnum > 0.0f)
        {
            //result_color.x() += kd.x() * (light.intensity.x() / vlight.squaredNorm())*cosnum;
            //result_color.y() += kd.y() * (light.intensity.y() / vlight.squaredNorm())*cosnum;
            //result_color.z() += kd.z() * (light.intensity.z() / vlight.squaredNorm())*cosnum;
            result_color+=kd.cwiseProduct((light.intensity / vlight.squaredNorm())*cosnum);
        }
        Eigen::Vector3f veye= eye_pos - point;
        //高光 ka*(I/r^2)*max(0,pow(normal(单位向量)点乘半程向量(单位向量),p))
        float halfcross=std::pow(std::max(0.0f, normal.dot((veye+vlight).normalized())),p);
        if(halfcross > 0.0f)
        {
            //result_color.x()+=ks.x() * (light.intensity.x() / vlight.squaredNorm())*halfcross;
            //result_color.y()+=ks.y() * (light.intensity.y() / vlight.squaredNorm())*halfcross;
            //result_color.z()+=ks.z() * (light.intensity.z() / vlight.squaredNorm())*halfcross;
            result_color+=ks.cwiseProduct((light.intensity / vlight.squaredNorm())*halfcross);
        }
        //环境光 Ka*Ia
        result_color+=ka.cwiseProduct(amb_light_intensity);


        // TODO: For each light source in the code, calculate what the *ambient*, *diffuse*, and *specular*
        // components are. Then, accumulate that result on the *result_color* object.
    }
    return result_color * 255.f;
}

Eigen::Vector3f displacement_fragment_shader(const fragment_shader_payload &payload)
{

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;
    // TODO: Implement displacement mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Position p = p + kn * n * h(u,v)
    // Normal n = normalize(TBN * ln)
    
    float x=normal.x();
    float y=normal.y();
    float z=normal.z();
    float u = payload.tex_coords.x();
	float v = payload.tex_coords.y();
	float w = payload.texture->width;
	float h = payload.texture->height;
    //这部分没看懂 只知道通过原法线做z轴构建一个坐标系 把u v关联性建立起来
    Eigen::Vector3f t={x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z)};
    Eigen::Vector3f b=normal.cross(t);
    Eigen::Matrix3f TBN;
    //构造坐标系
    TBN<<t.x(),b.x(),x,
        t.y(),b.y(),y,
        t.z(),b.z(),z;
    //c = kh*kn du dv计算 坐标+1求导
    float du=kh*kn*(payload.texture->getColor(u+1/w,v).norm()-payload.texture->getColor(u,v).norm());
    float dv=kh*kn*(payload.texture->getColor(u,v+1/h).norm()-payload.texture->getColor(u,v).norm());
    Eigen::Vector3f ln={-du,-dv,1.0};//切线逆时针旋转90度 计算新法线
    point += (kn * normal * payload.texture->getColor(u , v).norm());
    normal = (TBN * ln).normalized();
    //计算uv下新法线

    Eigen::Vector3f result_color = {0, 0, 0};

    for (auto &light : lights)
    {
        Eigen::Vector3f vlight = light.position - point;

        float cosnum = std::max(0.0f, normal.dot(vlight.normalized()));
        if (cosnum > 0.0f)
        {
            //result_color.x() += kd.x() * (light.intensity.x() / vlight.squaredNorm())*cosnum;
            //result_color.y() += kd.y() * (light.intensity.y() / vlight.squaredNorm())*cosnum;
            //result_color.z() += kd.z() * (light.intensity.z() / vlight.squaredNorm())*cosnum;
            result_color+=kd.cwiseProduct((light.intensity / vlight.squaredNorm())*cosnum);
        }
        Eigen::Vector3f veye= eye_pos - point;
        float halfcross=std::pow(std::max(0.0f, normal.dot((veye+vlight).normalized())),p);
        if(halfcross > 0.0f)
        {
            //result_color.x()+=ks.x() * (light.intensity.x() / vlight.squaredNorm())*halfcross;
            //result_color.y()+=ks.y() * (light.intensity.y() / vlight.squaredNorm())*halfcross;
            //result_color.z()+=ks.z() * (light.intensity.z() / vlight.squaredNorm())*halfcross;
            result_color+=ks.cwiseProduct((light.intensity / vlight.squaredNorm())*halfcross);
        }
        result_color+=ka.cwiseProduct(amb_light_intensity);
    }

    return result_color * 255.f;
}

Eigen::Vector3f bump_fragment_shader(const fragment_shader_payload &payload)
{

    Eigen::Vector3f ka = Eigen::Vector3f(0.005, 0.005, 0.005);
    Eigen::Vector3f kd = payload.color;
    Eigen::Vector3f ks = Eigen::Vector3f(0.7937, 0.7937, 0.7937);

    auto l1 = light{{20, 20, 20}, {500, 500, 500}};
    auto l2 = light{{-20, 20, 0}, {500, 500, 500}};

    std::vector<light> lights = {l1, l2};
    Eigen::Vector3f amb_light_intensity{10, 10, 10};
    Eigen::Vector3f eye_pos{0, 0, 10};

    float p = 150;

    Eigen::Vector3f color = payload.color;
    Eigen::Vector3f point = payload.view_pos;
    Eigen::Vector3f normal = payload.normal;

    float kh = 0.2, kn = 0.1;

    float x=normal.x();
    float y=normal.y();
    float z=normal.z();
    float u = payload.tex_coords.x();
	float v = payload.tex_coords.y();
	float w = payload.texture->width;
	float h = payload.texture->height;
    Eigen::Vector3f t={x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z)};
    Eigen::Vector3f b=normal.cross(t);
    Eigen::Matrix3f TBN;
    
    TBN<<t.x(),b.x(),x,
        t.y(),b.y(),y,
        t.z(),b.z(),z;
    
    float du=kh*kn*(payload.texture->getColor(u+1/w,v).norm()-payload.texture->getColor(u,v).norm());
    float dv=kh*kn*(payload.texture->getColor(u,v+1/h).norm()-payload.texture->getColor(u,v).norm());
    Eigen::Vector3f ln={-du,-dv,1.0};
    normal = (TBN * ln).normalized();
    // TODO: Implement bump mapping here
    // Let n = normal = (x, y, z)
    // Vector t = (x*y/sqrt(x*x+z*z),sqrt(x*x+z*z),z*y/sqrt(x*x+z*z))
    // Vector b = n cross product t
    // Matrix TBN = [t b n]
    // dU = kh * kn * (h(u+1/w,v)-h(u,v))
    // dV = kh * kn * (h(u,v+1/h)-h(u,v))
    // Vector ln = (-dU, -dV, 1)
    // Normal n = normalize(TBN * ln)


    Eigen::Vector3f result_color = {0, 0, 0};
    result_color = normal;

    return result_color * 255.f;
}

int main(int argc, const char **argv)
{
    std::vector<Triangle *> TriangleList;

    float angle = 140.0;
    bool command_line = false;

    std::string filename = "output.png";
    objl::Loader Loader;
    std::string obj_path = "../../models/spot/";

    // Load .obj File
    bool loadout = Loader.LoadFile("../../models/spot/spot_triangulated_good.obj");
    for (auto mesh : Loader.LoadedMeshes)
    {
        for (int i = 0; i < mesh.Vertices.size(); i += 3)
        {
            Triangle *t = new Triangle();
            for (int j = 0; j < 3; j++)
            {
                t->setVertex(j, Vector4f(mesh.Vertices[i + j].Position.X, mesh.Vertices[i + j].Position.Y, mesh.Vertices[i + j].Position.Z, 1.0));
                t->setNormal(j, Vector3f(mesh.Vertices[i + j].Normal.X, mesh.Vertices[i + j].Normal.Y, mesh.Vertices[i + j].Normal.Z));
                t->setTexCoord(j, Vector2f(mesh.Vertices[i + j].TextureCoordinate.X, mesh.Vertices[i + j].TextureCoordinate.Y));
            }
            TriangleList.push_back(t);
        }
    }

    rst::rasterizer r(700, 700);

    auto texture_path = "spot_texture.png";
    r.set_texture(Texture(obj_path + texture_path));

    std::function<Eigen::Vector3f(fragment_shader_payload)> active_shader = texture_fragment_shader;

    if (argc >= 2)
    {
        command_line = true;
        filename = std::string(argv[1]);

        if (argc == 3 && std::string(argv[2]) == "texture")
        {
            std::cout << "Rasterizing using the texture shader\n";
            active_shader = texture_fragment_shader;
            texture_path = "spot_texture.png";
            r.set_texture(Texture(obj_path + texture_path));
        }
        else if (argc == 3 && std::string(argv[2]) == "normal")
        {
            std::cout << "Rasterizing using the normal shader\n";
            active_shader = normal_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "phong")
        {
            std::cout << "Rasterizing using the phong shader\n";
            active_shader = phong_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "bump")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = bump_fragment_shader;
        }
        else if (argc == 3 && std::string(argv[2]) == "displacement")
        {
            std::cout << "Rasterizing using the bump shader\n";
            active_shader = displacement_fragment_shader;
        }
    }

    Eigen::Vector3f eye_pos = {0, 0, 10};

    r.set_vertex_shader(vertex_shader);
    r.set_fragment_shader(active_shader);

    int key = 0;
    int frame_count = 0;

    if (command_line)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);
        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imwrite(filename, image);

        return 0;
    }

    while (key != 27)
    {
        r.clear(rst::Buffers::Color | rst::Buffers::Depth);

        r.set_model(get_model_matrix(angle));
        r.set_view(get_view_matrix(eye_pos));
        r.set_projection(get_projection_matrix(45.0, 1, 0.1, 50));

        // r.draw(pos_id, ind_id, col_id, rst::Primitive::Triangle);
        r.draw(TriangleList);
        cv::Mat image(700, 700, CV_32FC3, r.frame_buffer().data());
        image.convertTo(image, CV_8UC3, 1.0f);
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);

        cv::imshow("image", image);
        cv::imwrite(filename, image);
        key = cv::waitKey(10);
        angle+=30.0;
        if (key == 'a')
        {
            angle -= 0.1;
        }
        else if (key == 'd')
        {
            angle += 0.1;
        }
    }
    return 0;
}
