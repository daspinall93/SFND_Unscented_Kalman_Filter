#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd::Zero(5);

  // initial covariance matrix
  P_ = MatrixXd::Zero(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;
  
  /**
   * DO NOT MODIFY measurement noise values below.
   * These are provided by the sensor manufacturer.
   */

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;
  
  /**
   * End DO NOT MODIFY section for measurement noise values 
   */
  
  /**
   * TODO: Complete the initialization. See ukf.h for other member properties.
   * Hint: one or more values initialized above might be wildly off...
   */
  n_x_ = 5;
  n_aug_ = 7;
  lambda_ = 3 - n_aug_;
  time_us_ = 0.01;
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);

  // Initialise weights
  weights_ = VectorXd(2 * n_aug_ + 1);
  double weight_0 = lambda_/(lambda_ + n_aug_);
  double weight = 0.5/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i=1; i < 2 * n_aug_ + 1; ++i) 
  {  
    weights_(i) = weight;
  }

  is_initialized_ = false;
}

UKF::~UKF() {}

void UKF::ProcessMeasurement(MeasurementPackage meas_package) 
{
  /**
   * TODO: Complete this function! Make sure you switch between lidar and radar
   * measurements.
   */

  if (!is_initialized_)
  {
    // Initialise the state vector and uncertainty matrix.
    if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    {
      auto rho = meas_package.raw_measurements_[0];
      auto phi = meas_package.raw_measurements_[1];
      auto rhoRate = meas_package.raw_measurements_[2];

      auto posX = rho * sin(phi);
      auto posY = rho * cos(phi);

      x_ << posX, posY, rhoRate, phi, 0;
      P_.setIdentity();
    }
    else if(meas_package.sensor_type_ == MeasurementPackage::LASER)
    {
      auto x = meas_package.raw_measurements_[0];
      auto y = meas_package.raw_measurements_[1];

      x_ << x, y, 0, 0, 0;

      P_.setIdentity();
    }
    else
    {
      std::cerr << "Incorrect sensor type. Program will be terminated" << std::endl;
      exit(-1);
    }
    is_initialized_ = true;
    time_us_ = meas_package.timestamp_;
  }

  // Carry out prediction
  double dt = (meas_package.timestamp_ - time_us_) / 1e6;
  time_us_ = meas_package.timestamp_;  
  Prediction(dt);

  // Carry out measurment update
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR)
    UpdateRadar(meas_package);
  else if (meas_package.sensor_type_ == MeasurementPackage::LASER)
    UpdateLidar(meas_package);
  else
  {
    std::cerr << "Incorrect sensor type value.. EXITING" << std::endl;
    exit(-1);
  }
}

double wrapAngle(double angle)
{
    double wrappedAngle = fmod(angle + M_PI, 2.0 * M_PI);
    if (wrappedAngle < 0)
        wrappedAngle += 2.0 * M_PI;
    return wrappedAngle - M_PI;
}

void UKF::Prediction(double delta_t) 
{
  /**
   * TODO: Complete this function! Estimate the object's location. 
   * Modify the state vector, x_. Predict sigma points, the state, 
   * and the state covariance matrix.
   */

    MatrixXd augXSigma = GenAugSigmaPoints();

    // Predict sigma points
    for (int i = 0; i < 2 * n_aug_ + 1; ++i) 
    {
      double posX = augXSigma(0,i);
      double posY = augXSigma(1,i);
      double vel = augXSigma(2,i);
      double yaw = augXSigma(3,i);
      double yawRate = augXSigma(4,i);
      double nuAcc = augXSigma(5,i);
      double nuYawAcc = augXSigma(6,i);

      // Predict state values values
      double predPosX, predPosY;
      if (fabs(yawRate) > 0.001) 
      {
          predPosX = posX + vel/yawRate * ( sin(yaw + yawRate * delta_t) - sin(yaw));
          predPosY = posY + vel/yawRate * ( cos(yaw) - cos(yaw + yawRate * delta_t) );
      } 
      else
      {
          predPosX = posX + vel * delta_t * cos(yaw);
          predPosY = posY + vel * delta_t * sin(yaw);
      }
      predPosX = predPosX + 0.5 * nuAcc * delta_t * delta_t * cos(yaw);
      predPosY = predPosY + 0.5 * nuAcc * delta_t * delta_t * sin(yaw);

      double predVel = vel + nuAcc * delta_t;
      double predYaw = yaw + yawRate * delta_t + 0.5 * nuYawAcc * delta_t * delta_t;
      double predYawRate = yawRate + nuYawAcc * delta_t;

      // Store predicted values in state
      Xsig_pred_(0,i) = predPosX;
      Xsig_pred_(1,i) = predPosY;
      Xsig_pred_(2,i) = predVel;
      Xsig_pred_(3,i) = predYaw;
      Xsig_pred_(4,i) = predYawRate;
    }

    // Predict mean and covariance
    x_.fill(0.0);
    for (int i = 0; i < Xsig_pred_.cols(); i++) 
      x_ = x_ + weights_(i) * Xsig_pred_.col(i);

    P_.fill(0.0);
    for(int i = 0; i < Xsig_pred_.cols(); i++)
    {
      VectorXd diff = Xsig_pred_.col(i) - x_;
      diff(3) = wrapAngle(diff(3));

      P_ += weights_(i) * diff * diff.transpose();
    }
}

MatrixXd UKF::GenAugSigmaPoints()
{
  // Create augmented state vector
  VectorXd augX = VectorXd(n_aug_);
  augX.head(n_x_) = x_;
  augX(5) = 0;
  augX(6) = 0;

  // Create augmented covariance matrix
  MatrixXd augP = MatrixXd(n_aug_, n_aug_);
  augP.fill(0.0);
  MatrixXd Q = MatrixXd(2, 2);
  Q << std_a_ * std_a_, 0,
        0, std_yawdd_ * std_yawdd_;
  augP.topLeftCorner(n_x_, n_x_) = P_;
  augP.bottomRightCorner(2, 2) = Q;

  // create square root matrix
  MatrixXd augA = augP.llt().matrixL();
  auto sqrtMatrix = sqrt((lambda_ + n_aug_)) * augA;

  // Create augmented sigma points
  MatrixXd augXSigma = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  augXSigma.col(0) = augX;
  auto positiveSigmaPoints = sqrtMatrix.colwise() + augX;
  auto negativeSigmaPoints = (-1 * sqrtMatrix).colwise() + augX;
  for(int i = 0; i < positiveSigmaPoints.cols(); i++)
  {
    augXSigma.col(i + 1) = positiveSigmaPoints.col(i);
    augXSigma.col(i + 1 + n_aug_) = negativeSigmaPoints.col(i);
  }

  return augXSigma;
}

void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use lidar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the lidar NIS, if desired.
   */

  int nMeasDim = 2;

  // Calculate measurement uncertainty
  MatrixXd R = MatrixXd::Zero(nMeasDim, nMeasDim);
  R(0, 0) = std_laspx_ * std_laspx_;
  R(1, 1) = std_laspy_ * std_laspy_;

  // Put sigma points in measurement space
  MatrixXd sigZ = MatrixXd::Zero(nMeasDim, 2 * n_aug_ + 1);
  for(auto i = 0; i < Xsig_pred_.cols(); i++)
  {
    sigZ(0, i) = Xsig_pred_(0, i);
    sigZ(1, i) = Xsig_pred_(1, i);
  }

  // Calculate predicted measurement
  VectorXd predZ = VectorXd::Zero(nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
    predZ += weights_(i) * sigZ.col(i);

  // Calculate innovation matrix
  MatrixXd S = MatrixXd::Zero(nMeasDim, nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
  {
    VectorXd innovation = sigZ.col(i) - predZ;
    S += weights_(i) * innovation * innovation.transpose();

    innovation(1) = wrapAngle(innovation(1));
  }
  S = S + R;

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
  {
    VectorXd diffX = Xsig_pred_.col(i) - x_;
    VectorXd diffZ = sigZ.col(i) - predZ;
    wrapAngle(diffX(3));
    wrapAngle(diffZ(1));

    Tc += weights_(i) * diffX * diffZ.transpose();
  }

  MatrixXd K = Tc * S.inverse();  

  // Update state and mean covariance
  VectorXd innovation = meas_package.raw_measurements_ - predZ;  
  x_ = x_ + K * innovation;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  double lidarNIS = innovation.transpose() * S.inverse() * innovation;
  std::cout << "Lidar NIS = " << lidarNIS << std::endl;
}

void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
   * TODO: Complete this function! Use radar data to update the belief 
   * about the object's position. Modify the state vector, x_, and 
   * covariance, P_.
   * You can also calculate the radar NIS, if desired.
   */

  int nMeasDim = 3;

  // Measurement uncertainty
  MatrixXd R = MatrixXd::Zero(nMeasDim, nMeasDim);
  R(0, 0) = std_radr_ * std_radr_;
  R(1, 1) = std_radphi_ * std_radphi_;
  R(2, 2) = std_radrd_ * std_radrd_;

  // Put sigma points in measurement space
  MatrixXd sigZ = MatrixXd::Zero(nMeasDim, 2 * n_aug_ + 1);
  for(auto i = 0; i < Xsig_pred_.cols(); i++)
  {
    auto posX  = Xsig_pred_(0, i);
    auto posY  = Xsig_pred_(1, i);
    auto vel   = Xsig_pred_(2, i);
    auto psi = Xsig_pred_(3, i);

    auto range = sqrt(posX * posX + posY * posY);
    auto phi = atan2(posY, posX);
    auto delta_phi = (posX * cos(psi) * vel + posY * sin(psi) * vel) / range;

    sigZ(0, i) = range;
    sigZ(1, i) = phi;
    sigZ(2, i) = delta_phi;
  }

  // Calculate predicted measurement
  VectorXd predZ = VectorXd::Zero(nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
    predZ += weights_(i) * sigZ.col(i);

  // Calculate innovation matrix
  MatrixXd S = MatrixXd::Zero(nMeasDim,nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
  {
    VectorXd innovation = sigZ.col(i) - predZ;
    innovation(1) = wrapAngle(innovation(1));

    S += weights_(i) * innovation * innovation.transpose();
  }
  S = S + R;

  // Calculate cross correlation matrix
  MatrixXd Tc = MatrixXd::Zero(n_x_, nMeasDim);
  for(auto i = 0; i < sigZ.cols(); i++)
  {
    VectorXd diffX = Xsig_pred_.col(i) - x_;
    VectorXd diffZ = sigZ.col(i) - predZ;
    diffX(3) = wrapAngle(diffX(3));
    diffZ(1) = wrapAngle(diffZ(1));

    Tc += weights_(i) * diffX * diffZ.transpose();
  }  

  MatrixXd K = Tc * S.inverse();  

  // Update state and mean covariance
  VectorXd innovation = meas_package.raw_measurements_ - predZ;  
  innovation(1) = wrapAngle(innovation(1));
  x_ = x_ + K * innovation;
  P_ = P_ - K * S * K.transpose();

  // Calculate NIS
  double radarNIS = innovation.transpose() * S.inverse() * innovation;
  std::cout << "Radar NIS " << radarNIS << std::endl;
}