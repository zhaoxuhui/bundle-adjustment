#include <iostream>
#include <g2o/core/base_vertex.h>//顶点类型
#include <g2o/core/base_unary_edge.h>//一元边类型
#include <g2o/core/block_solver.h>//求解器的实现,主要来自choldmod, csparse
#include <g2o/core/optimization_algorithm_levenberg.h>//列文伯格－马夸尔特
#include <g2o/core/optimization_algorithm_gauss_newton.h>//高斯牛顿法
#include <g2o/core/optimization_algorithm_dogleg.h>//Dogleg（狗腿方法）
#include <g2o/solvers/dense/linear_solver_dense.h>
#include <Eigen/Core>//矩阵库
#include <opencv2/core/core.hpp>//opencv2
#include <cmath>//数学库
#include <chrono>//时间库

using namespace std;
      
//定义曲线模型顶点，这也是我们的待优化变量
//这个顶点类继承于基础顶点类，基础顶点类是个模板类，模板参数表示优化变量的维度和数据类型
class CurveFittingVertex: public g2o::BaseVertex<4, Eigen::Vector4d>
{
public:
  //这在前面说过了，因为类中含有Eigen对象，为了防止内存不对齐，加上这句宏定义
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  
  //在基类中这是个=0的虚函数，所以必须重新定义
  //它用于重置顶点，例如全部重置为0
  virtual void setToOriginImpl()
  {
    //这里的_estimate是基类中的变量，直接用了
    _estimate << 0,0,0,0;
  }
  
  //这也是一个必须要重定义的虚函数
  //它的用途是对顶点进行更新，对应优化中X(k+1)=Xk+Dx
  //需要注意的是虽然这里的更新是个普通的加法，但并不是所有情况都是这样
  //例如相机的位姿，其本身没有加法运算
  //这时就需要我们自己定义"增量如何加到现有估计上"这样的算法了
  //这也就是为什么g2o没有为我们写好的原因
  virtual void oplusImpl( const double* update )
  {
    //注意这里的Vector4d，d是double的意思，f是float的意思
    _estimate += Eigen::Vector4d(update);
  }
  
  //虚函数  存盘和读盘：留空
  virtual bool read( istream& in ) {}
  virtual bool write( ostream& out ) const {}
};
      
      
//误差模型—— 曲线模型的边 
//模板参数：观测值维度(输入的参数维度)，类型，连接顶点类型(创建的顶点)   
class CurveFittingEdge: public g2o::BaseUnaryEdge<1,double,CurveFittingVertex>
{
public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double _x;
  
  //这种写法应该有一种似曾相识的感觉
  //在Ceres里定义代价函数结构体的时候，那里的构造函数也用了这种写法
  CurveFittingEdge( double x ): BaseUnaryEdge(), _x(x) {}

  //计算曲线模型误差 
  void computeError()
  {
    //顶点，用到了编译时类型检查
    //_vertices是基类中的成员变量
    const CurveFittingVertex* v = static_cast<const CurveFittingVertex*>(_vertices[0]);
    //获取顶点的优化变量
    const Eigen::Vector4d abcd = v->estimate();
    //_error、_measurement都是基类中的成员变量
    _error(0,0) = _measurement - std::exp(abcd(0,0)*_x*_x*_x + abcd(1,0)*_x*_x + abcd(2,0)*_x + abcd(3,0));
  }
  
  //存盘和读盘：留空
  virtual bool read( istream& in ) {}
  virtual bool write( ostream& out ) const {}
};
      
int main0( int argc, char** argv )
{
  //待估计函数为y=exp(3.5x^3+1.6x^2+0.3x+7.8)
  double a=3.5, b=1.6, c=0.3, d=7.8;
  int N=100;
  double w_sigma=1.0;
  cv::RNG rng;
  
  vector<double> x_data, y_data;
  
  cout<<"generating data: "<<endl;
  for (int i=0; i<N; i++)
  {
    double x = i/100.0;
    x_data.push_back (x);
    y_data.push_back (exp(a*x*x*x + b*x*x + c*x + d) + rng.gaussian (w_sigma));
    cout<<x_data[i]<<"\t"<<y_data[i]<<endl;
  }
  
  //构建图优化，先设定g2o
  //矩阵块：每个误差项优化变量维度为4，误差值维度为1
  typedef g2o::BlockSolver<g2o::BlockSolverTraits<4,1>> Block;

  // Step1 选择一个线性方程求解器
  std::unique_ptr<Block::LinearSolverType> linearSolver (new g2o::LinearSolverDense<Block::PoseMatrixType>());
  // Step2 选择一个稀疏矩阵块求解器
  std::unique_ptr<Block> solver_ptr (new Block(std::move(linearSolver)));
  // Step3 选择一个梯度下降方法，从GN、LM、DogLeg中选
  g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg ( std::move(solver_ptr));
  //g2o::OptimizationAlgorithmGaussNewton* solver = new g2o::OptimizationAlgorithmGaussNewton( std::move(solver_ptr));
  //g2o::OptimizationAlgorithmDogleg* solver = new g2o::OptimizationAlgorithmDogleg(std::move(solver_ptr));

  //稀疏优化模型
  g2o::SparseOptimizer optimizer;
  //设置求解器
  optimizer.setAlgorithm(solver);
  //打开调试输出
  optimizer.setVerbose(true);
          
  //往图中增加顶点
  CurveFittingVertex* v = new CurveFittingVertex();
  //这里的(0,0,0,0)可以理解为顶点的初值了，不同的初值会导致迭代次数不同，可以试试
  v->setEstimate(Eigen::Vector4d(0,0,0,0));
  v->setId(0);
  optimizer.addVertex(v);

  //往图中增加边
  for (int i=0; i<N; i++)
  {
    //新建边带入观测数据
    CurveFittingEdge* edge = new CurveFittingEdge(x_data[i]);
    edge->setId(i);
    //设置连接的顶点，注意使用方式
    edge->setVertex(0, v);
    //观测数值
    edge->setMeasurement(y_data[i]);
    //信息矩阵：协方差矩阵之逆，这里各边权重相同。这里Eigen的Matrix其实也是模板类
    edge->setInformation(Eigen::Matrix<double,1,1>::Identity()*1/(w_sigma*w_sigma));
    optimizer.addEdge(edge);
  }
  
  //执行优化
  cout<<"start optimization"<<endl;
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时
  
  //初始化优化器
  optimizer.initializeOptimization();
  //优化次数
  optimizer.optimize(100);
  
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//结束计时
  
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
  cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
          
  //输出优化值
  Eigen::Vector4d abc_estimate = v->estimate();
  cout<<"estimated model: "<<abc_estimate.transpose()<<endl;
  
  return 0;
}