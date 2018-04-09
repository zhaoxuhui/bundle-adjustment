#include<iostream>
#include"ORBMatching.h"
#include<g2o/core/sparse_optimizer.h>
#include<g2o/core/block_solver.h>
#include<g2o/core/robust_kernel.h>
#include<g2o/core/robust_kernel_impl.h>
#include<g2o/core/optimization_algorithm_levenberg.h>
#include<g2o/solvers/cholmod/linear_solver_cholmod.h>
#include<g2o/types/slam3d/se3quat.h>
#include<g2o/types/sba/types_six_dof_expmap.h>
#include <chrono>

using namespace std;

void epipolar(vector<Point2f> pts1, vector<Point2f> pts2, double fx, double fy, double dx, double dy) {

    cout << "--------epipolar constraint--------" << endl;
    Point2d principal_point(dx, dy);
    Mat essential_matrix;
    double f = (fx + fy) / 2.0;
    essential_matrix = findEssentialMat(pts1, pts2, f, principal_point, RANSAC);
    Mat R, t;
    recoverPose(essential_matrix, pts1, pts2, R, t, f, principal_point);
    cout << "R:" << endl << R << endl;
    cout << "t:" << endl << t << endl;
    cout << "--------epipolar constraint--------" << endl;
    return;
}

void graph_optimization(vector<Point2f> &pts1, vector<Point2f> &pts2, double fx, double fy, double cx, double cy) {
    cout << "--------bundle adjustment--------" << endl;

    //矩阵块：每个误差项优化变量维度为6，误差值维度为3
    typedef g2o::BlockSolver<g2o::BlockSolverTraits<6, 3>> Block;
    // Step1 选择一个线性方程求解器
    std::unique_ptr<Block::LinearSolverType> linearSolver(new g2o::LinearSolverCholmod<Block::PoseMatrixType>());
    // Step2 选择一个稀疏矩阵块求解器
    std::unique_ptr<Block> solver_ptr(new Block(std::move(linearSolver)));
    // Step3 选择一个梯度下降方法，从GN、LM、DogLeg中选
    g2o::OptimizationAlgorithmLevenberg *solver = new g2o::OptimizationAlgorithmLevenberg(std::move(solver_ptr));

    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    //添加两个位姿节点，默认值为单位Pose，并且将第一个位姿节点也就是第一帧的位姿固定
    for (int i = 0; i < 2; i++) {
        g2o::VertexSE3Expmap *v = new g2o::VertexSE3Expmap();
        v->setId(i);
        if (i == 0) {
            v->setFixed(true);
        }
        v->setEstimate(g2o::SE3Quat());
        optimizer.addVertex(v);
    }

    //添加特征点作为节点
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::VertexSBAPointXYZ *v = new g2o::VertexSBAPointXYZ();
        v->setId(2 + i);
        //深度未知，设为1，利用相机模型，相当于是在求归一化相机坐标
        double z = 1;
        double x = (pts1[i].x - cx) * z / fx;
        double y = (pts1[i].y - cy) * z / fy;
        v->setMarginalized(true);
        v->setEstimate(Eigen::Vector3d(x, y, z));
        optimizer.addVertex(v);
    }

    //相机内参
    g2o::CameraParameters *camera = new g2o::CameraParameters(fx, Eigen::Vector2d(cx, cy), 0);
    camera->setId(0);
    optimizer.addParameter(camera);

    //添加第一帧中的边
    vector<g2o::EdgeProjectXYZ2UV *> edges;
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(0)));

        edge->setMeasurement(Eigen::Vector2d(pts1[i].x, pts1[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());

        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    //添加第二帧中的边
    for (size_t i = 0; i < pts2.size(); i++) {
        g2o::EdgeProjectXYZ2UV *edge = new g2o::EdgeProjectXYZ2UV();

        edge->setVertex(0, dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2)));
        edge->setVertex(1, dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1)));

        edge->setMeasurement(Eigen::Vector2d(pts2[i].x, pts2[i].y));
        edge->setInformation(Eigen::Matrix2d::Identity());
        edge->setParameterId(0, 0);
        edge->setRobustKernel(new g2o::RobustKernelHuber());
        optimizer.addEdge(edge);
        edges.push_back(edge);
    }

    cout << "开始优化" << endl;
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时
    optimizer.initializeOptimization();
    optimizer.optimize(10);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//结束计时
    cout << "优化完毕" << endl;

    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;

    //输出变换矩阵T
    g2o::VertexSE3Expmap *v = dynamic_cast<g2o::VertexSE3Expmap *>(optimizer.vertex(1));
    Eigen::Isometry3d pose = v->estimate();
    cout << "Pose=" << endl << pose.matrix() << endl;

    //优化后所有特征点的位置
    for (size_t i = 0; i < pts1.size(); i++) {
        g2o::VertexSBAPointXYZ *v = dynamic_cast<g2o::VertexSBAPointXYZ *>(optimizer.vertex(i + 2));
        cout << "vertex id:" << i + 2 << ",pos=";
        Eigen::Vector3d pos = v->estimate();
        cout << pos(0) << "," << pos(1) << "," << pos(2) << endl;
    }

    //估计inliner的个数
    int inliners = 0;
    for (auto e:edges) {
        e->computeError();
        //chi2()如果很大，说明此边的值与其它边很不相符
        if (e->chi2() > 1) {
            cout << "error = " << e->chi2() << endl;
        } else {
            inliners++;
        }
    }
    cout << "inliners in total points:" << inliners << "/" << pts1.size() + pts2.size() << endl;
    optimizer.save("ba.g2o");
    cout << "--------bundle adjustment--------" << endl;
    return;
};

double cx = 1535.89;
double cy = 2046.97;
double fx = 1536.2;
double fy = 1536.2;
string img_path1 = "/root/satellite_slam_data/tianjin/jpg2/VBZ1_201801051905_002_0002_L1A.jpg";
string img_path2 = "/root/satellite_slam_data/tianjin/jpg2/VBZ1_201801051905_002_0003_L1A.jpg";


int main(int argc, char **argv) {

    vector<Point2f> pts1, pts2;
    getMatchPoints(img_path1, img_path2, pts1, pts2);
    graph_optimization(pts1, pts2, fx, fy, cx, cy);
    epipolar(pts1, pts2, fx, fy, cx, cy);

    return 0;
}
