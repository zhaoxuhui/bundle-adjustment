//
// Created by root on 18-4-9.
//

#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/factory.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"

#include "g2o/types/slam2d/vertex_se2.h"
#include "g2o/types/slam3d/vertex_se3.h"

#include <iostream>

//G2O_USE_TYPE_GROUP保证能保证graph的load和save的成功
//也就是optimizer.load和optimizer.save在执行时有相对的类型对应.
//slam3d和slam2d对应不同的顶点和边的类型.这里都registration.
G2O_USE_TYPE_GROUP(slam3d);
G2O_USE_TYPE_GROUP(slam2d);

using namespace std;
using namespace g2o;

#define MAXITERATION 10


int main2(){

    char* path = "manhattanOlson3500.g2o";

    unique_ptr<BlockSolverX::LinearSolverType> linearSolver(new LinearSolverCholmod<BlockSolverX ::PoseMatrixType>());
    unique_ptr<BlockSolverX> blockSolverX(new BlockSolverX(move(linearSolver)));
    OptimizationAlgorithmLevenberg* solver = new OptimizationAlgorithmLevenberg(move(blockSolverX));

    SparseOptimizer optimizer;

    if(!optimizer.load(path)){
        cout<<"Error loading file."<<endl;
        return -1;
    }else{
        cout<<"Loaded "<<optimizer.vertices().size()<<" vertices"<<endl;
        cout<<"Loaded "<<optimizer.edges().size()<<" edges."<<endl;
    }

    //这里这样做的目的是固定第一个点的位姿，当然不固定也可以，但最终优化的效果会有差别
    //for sphere2500
    //VertexSE3* firstRobotPose = dynamic_cast<VertexSE3*>(optimizer.vertex(0));
    //for manhattanOlson3500
    VertexSE2* firstRobotPose = dynamic_cast<VertexSE2*>(optimizer.vertex(0));
    firstRobotPose->setFixed(true);

    optimizer.setAlgorithm(solver);
    optimizer.setVerbose(true);

    cout<<"开始优化..."<<endl;
    optimizer.initializeOptimization();
    optimizer.optimize(MAXITERATION);
    cout<<"优化完成..."<<endl;
    optimizer.save("after.g2o");
    optimizer.clear();

    return 0;
}