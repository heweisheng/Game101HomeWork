// clang-format off
//
// Created by goksu on 4/6/19.
//

#include <algorithm>
#include <vector>
#include "rasterizer.hpp"
#include <opencv2/opencv.hpp>
#include <math.h>
//超采样抗锯齿
#define MSAAW 2.0f
#define MSAAH 2.0f 

rst::pos_buf_id rst::rasterizer::load_positions(const std::vector<Eigen::Vector3f> &positions)
{
    auto id = get_next_id();
    pos_buf.emplace(id, positions);

    return {id};
}

rst::ind_buf_id rst::rasterizer::load_indices(const std::vector<Eigen::Vector3i> &indices)
{
    auto id = get_next_id();
    ind_buf.emplace(id, indices);

    return {id};
}

rst::col_buf_id rst::rasterizer::load_colors(const std::vector<Eigen::Vector3f> &cols)
{
    auto id = get_next_id();
    col_buf.emplace(id, cols);

    return {id};
}

auto to_vec4(const Eigen::Vector3f& v3, float w = 1.0f)
{
    return Vector4f(v3.x(), v3.y(), v3.z(), w);
}


static bool insideTriangle(float x, float y, const Vector3f* _v)
{   
    Vector2f checkpoint={x,y};
    Vector2f l1=(_v[0]-_v[1]).head<2>();
    Vector2f l2=(_v[1]-_v[2]).head<2>();
    Vector2f l3=(_v[2]-_v[0]).head<2>();
    Vector2f p1=(_v[1].head<2>()-checkpoint);
    Vector2f p2=(_v[2].head<2>()-checkpoint);
    Vector2f p3=(_v[0].head<2>()-checkpoint);
    float c1 = l1.x()*p1.y()-l1.y()*p1.x();
    float c2 = l2.x()*p2.y()-l2.y()*p2.x();
    float c3 = l3.x()*p3.y()-l3.y()*p3.x();
    //用向量的叉乘结果判定，如果都在同一侧(全正，全负 则在范围中)
    if(c1<0.0&&c2<0.0&&c3<0.0)
        return true;
    else if(c1>=0.0&&c2>=0.0&&c3>=0.0)
        return true;

    return false;
    // TODO : Implement this function to check if the point (x, y) is inside the triangle represented by _v[0], _v[1], _v[2]
}

static std::tuple<float, float, float> computeBarycentric2D(float x, float y, const Vector3f* v)
{
    float c1 = (x*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*y + v[1].x()*v[2].y() - v[2].x()*v[1].y()) / (v[0].x()*(v[1].y() - v[2].y()) + (v[2].x() - v[1].x())*v[0].y() + v[1].x()*v[2].y() - v[2].x()*v[1].y());
    float c2 = (x*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*y + v[2].x()*v[0].y() - v[0].x()*v[2].y()) / (v[1].x()*(v[2].y() - v[0].y()) + (v[0].x() - v[2].x())*v[1].y() + v[2].x()*v[0].y() - v[0].x()*v[2].y());
    float c3 = (x*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*y + v[0].x()*v[1].y() - v[1].x()*v[0].y()) / (v[2].x()*(v[0].y() - v[1].y()) + (v[1].x() - v[0].x())*v[2].y() + v[0].x()*v[1].y() - v[1].x()*v[0].y());
    return {c1,c2,c3};
}

void rst::rasterizer::draw(pos_buf_id pos_buffer, ind_buf_id ind_buffer, col_buf_id col_buffer, Primitive type)
{
    auto& buf = pos_buf[pos_buffer.pos_id];
    auto& ind = ind_buf[ind_buffer.ind_id];
    auto& col = col_buf[col_buffer.col_id];

    float f1 = (50 - 0.1) / 2.0;
    float f2 = (50 + 0.1) / 2.0;

    Eigen::Matrix4f mvp = projection * view * model;
    for (auto& i : ind)
    {
        Triangle t;
        Eigen::Vector4f v[] = {
                mvp * to_vec4(buf[i[0]], 1.0f),
                mvp * to_vec4(buf[i[1]], 1.0f),
                mvp * to_vec4(buf[i[2]], 1.0f)
        };
        //Homogeneous division
        for (auto& vec : v) {
            vec /= vec.w();
        }
        //Viewport transformation
        for (auto & vert : v)
        {
            vert.x() = 0.5*width*(vert.x()+1.0);
            vert.y() = 0.5*height*(vert.y()+1.0);
            vert.z() = vert.z() * f1 + f2;
        }

        for (int i = 0; i < 3; ++i)
        {
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
            t.setVertex(i, v[i].head<3>());
        }

        auto col_x = col[i[0]];
        auto col_y = col[i[1]];
        auto col_z = col[i[2]];

        t.setColor(0, col_x[0], col_x[1], col_x[2]);
        t.setColor(1, col_y[0], col_y[1], col_y[2]);
        t.setColor(2, col_z[0], col_z[1], col_z[2]);

        rasterize_triangle(t);
    }
}

//做aabb包围盒 不知道怎么搞rect 用两个点表示矩阵范围
static std::array<Eigen::Vector2f,2> getboxaabb(std::array<Eigen::Vector4f, 3>& vertexs)
{
    std::array<Eigen::Vector2f,2> rect;
    Eigen::Vector2f& start=rect[0];
    Eigen::Vector2f& end=rect[1];
    start.x()=FLT_MAX;
    start.y()=FLT_MAX;
    end.x()=FLT_MIN;
    end.y()=FLT_MIN;
    //判定点所在位置在不在矩阵中，不在则扩大矩阵范围
    for(auto it:vertexs)
    {
        if(it.x()<start.x())
            start.x()=it.x();
        if(it.y()<start.y())
            start.y()=it.y();
        if(it.x()>end.x())
            end.x()=it.x();
        if(it.y()>end.y())
            end.y()=it.y();

    }
    return rect;
}

//Screen space rasterization
void rst::rasterizer::rasterize_triangle(const Triangle& t) {
    auto v = t.toVector4();
    //去掉齐次 转三维图
    std::array<Eigen::Vector3f,3> pt={v[0].head<3>(),v[1].head<3>(),v[2].head<3>()};
    auto box=getboxaabb(v);
    auto start=box[0];
    auto end=box[1];
    Eigen::Vector3f color=t.getColor();
    //只画包围盒内部
    for(int x=start.x();x<end.x();x++)
    {
        for(int y=start.y();y<end.y();y++)
        {
            int draw=0;
            //计算一个像素绘制的内容 这里可以提取成函数
            for(int i=0;i<MSAAW;i++)
            {
                for(int j=0;j<MSAAW;j++)
                {
                    if(!insideTriangle(x + i* 1.0f/MSAAW,y+ j *1.0f/MSAAH,&pt[0]))
                    {
                        continue;
                    }
                    //插值算法下面已经给出，直接使用
                    auto[alpha, beta, gamma] = computeBarycentric2D(x + i* 1.0f/MSAAW,y+ j *1.0f/MSAAH, t.v);
                    float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
                    float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
                    z_interpolated *= w_reciprocal;
                    //把深度图需要一起抗锯齿
                    auto ind = (y * MSAAH + j)* width * MSAAW + x * MSAAW + i  ;                    
                    if(depth_buf[ind]>z_interpolated)//深度结果是负数 初始化是无穷大 那值越小权重越大
                    {
                        depth_buf[ind]=z_interpolated;
                        draw++;
                        
                    }
                }
            }
            if(draw)
            {
                Eigen::Vector3f drawcolor = 
                {
                    color.x()*draw/MSAAW/MSAAH,
                    color.y()*draw/MSAAW/MSAAH,
                    color.z()*draw/MSAAW/MSAAH
                };

                Eigen::Vector3f drawpoint(x,y,1.0);
                set_pixel(drawpoint,drawcolor);
            }
        }
    }
    // TODO : Find out the bounding box of current triangle.
    // iterate through the pixel and find if the current pixel is inside the triangle

    // If so, use the following code to get the interpolated z value.
    //auto[alpha, beta, gamma] = computeBarycentric2D(x, y, t.v);
    //float w_reciprocal = 1.0/(alpha / v[0].w() + beta / v[1].w() + gamma / v[2].w());
    //float z_interpolated = alpha * v[0].z() / v[0].w() + beta * v[1].z() / v[1].w() + gamma * v[2].z() / v[2].w();
    //z_interpolated *= w_reciprocal;

    // TODO : set the current pixel (use the set_pixel function) to the color of the triangle (use getColor function) if it should be painted.
}

void rst::rasterizer::set_model(const Eigen::Matrix4f& m)
{
    model = m;
}

void rst::rasterizer::set_view(const Eigen::Matrix4f& v)
{
    view = v;
}

void rst::rasterizer::set_projection(const Eigen::Matrix4f& p)
{
    projection = p;
}

void rst::rasterizer::clear(rst::Buffers buff)
{
    if ((buff & rst::Buffers::Color) == rst::Buffers::Color)
    {
        std::fill(frame_buf.begin(), frame_buf.end(), Eigen::Vector3f{0, 0, 0});
    }
    if ((buff & rst::Buffers::Depth) == rst::Buffers::Depth)
    {
        std::fill(depth_buf.begin(), depth_buf.end(), std::numeric_limits<float>::infinity());
    }
}

rst::rasterizer::rasterizer(int w, int h) : width(w), height(h)
{
    frame_buf.resize(w * h);
    //改成超采样
    depth_buf.resize(w * h * MSAAW*MSAAH);
}

int rst::rasterizer::get_index(int x, int y)
{
    return (height-1-y)*width + x;
}

void rst::rasterizer::set_pixel(const Eigen::Vector3f& point, const Eigen::Vector3f& color)
{
    //old index: auto ind = point.y() + point.x() * width;
    auto ind = (height-1-point.y())*width + point.x();
    frame_buf[ind] = color;

}

// clang-format on