#include <iostream>
#include <fstream>
#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/contrib/contrib.hpp>

using namespace cv;
using namespace std;


static  Mat formatImagesForPCA(const vector<Mat> &data)
{
	//Mat dst(static_cast<int>(data.size()), data[0].rows*data[0].cols, CV_64FC1);
	size_t n = data.size();
	size_t d = data[0].total();
	int rtype = CV_64FC1;
	Mat result((int)n, (int)d, rtype);
	
	for(unsigned int i = 0; i < n; i++)
	{
		Mat row_i = result.row(i);
		if(data[i].isContinuous()) {
			data[i].reshape(1,1).convertTo(row_i, rtype, 1, 0);
		}
		else{
			data[i].clone().reshape(1, 1).convertTo(row_i, rtype, 1, 0);
		}
	}
	return result;
}

vector<Mat>  get_filelist_Train(const string& filepath, const int numImgs)
{
	vector<Mat> images;
	char * filename = new char[100];	
	for(int i=1;i<=numImgs;i++){		
		//create the file name of an image
		sprintf(filename,"faceRecogDatabase\\TrainDatabase\\%i.jpg",i);
		images.push_back(imread(filename, 0));
	}	
	return images;
}


vector<Mat>  get_filelist_Test(const string& filepath, const int numImgs)
{
	vector<Mat> images;
	char * filename = new char[100];	
	for(int i=1;i<=numImgs;i++){		
		//create the file name of an image
		sprintf(filename,"faceRecogDatabase\\TestDatabase\\%i.jpg",i);
		images.push_back(imread(filename, 0));
	}	
	return images;
}

void myPCA(Mat data, Mat &mean, Mat &eigenvectors, Mat &eigenvalues, int maxComponents)
{
	int covar_flags = CV_COVAR_SCALE;
	int i, len, in_count;
	Size mean_sz;

	CV_Assert( data.channels() == 1 );
	len = data.cols;
	in_count = data.rows;
	covar_flags |= CV_COVAR_ROWS;
	mean_sz = Size(len, 1);
	

	int count = std::min(len, in_count), out_count = count;
	if( maxComponents > 0 )
		out_count = std::min(count, maxComponents);

	if( len <= in_count )
		covar_flags |= CV_COVAR_NORMAL;

	int ctype = std::max(CV_32F, data.depth());
	mean.create( mean_sz, ctype );

	Mat covar( count, count, ctype );
	
	//计算协方差矩阵，均值
	calcCovarMatrix( data, covar, mean, covar_flags, ctype );
	
	//计算特征值，特征向量
	eigen( covar, eigenvalues, eigenvectors );

	//归一化
	if( !(covar_flags & CV_COVAR_NORMAL) )
	{
		Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
		if( data.type() != ctype || tmp_mean.data == mean.data )
		{
			data.convertTo( tmp_data, ctype );
			subtract( tmp_data, tmp_mean, tmp_data );
		}
		else
		{
			subtract( data, tmp_mean, tmp_mean );
			tmp_data = tmp_mean;
		}

		Mat evects1(count, len, ctype);
		gemm( eigenvectors, tmp_data, 1, Mat(), 0, evects1, 0);
		eigenvectors = evects1;

		// normalize eigenvectors
		for( i = 0; i < out_count; i++ )
		{
			Mat vec = eigenvectors.row(i);
			normalize(vec, vec);
		}
	}

	if( count > out_count )
	{
		// use clone() to physically copy the data and thus deallocate the original matrices
		eigenvalues = eigenvalues.rowRange(0,out_count).clone();
		eigenvectors = eigenvectors.rowRange(0,out_count).clone();
	}
}


///////////////////////
// Main
int main(int argc, char** argv)
{
	// vector to hold the images
	vector<Mat> imagesTrain, imagesTest;

	// Read in the data. This can fail if not valid
	imagesTrain = get_filelist_Train("faceRecogDatabase\\TestDatabase\\",280);
	imagesTest = get_filelist_Test("faceRecogDatabase\\TestDatabase\\",280);

	// Quit if there are not enough images for this demo.
	if(imagesTrain.size() <= 1) {
		string error_message = "This demo needs at least 2 images to work. Please add more images to your data set!";
		CV_Error(CV_StsError, error_message);
	}
	
	// ----------------------------------------------------------------------------------------------------------------------//
	// -------------------------------------------------------training------------------------------------------------------//
	// ----------------------------------------------------------------------------------------------------------------------//
	// Reshape and stack images into a rowMatrix
	Mat data = formatImagesForPCA(imagesTrain);

	// number of samples
	int n = data.rows;
	Mat _mean;// store the mean vector 得到均值脸
	Mat _eigenvalues; // eigenvalues by row 得到特征值
	Mat _eigenvectors; // eigenvectors by column 转置特征向量为列向量
	vector<Mat> _projections;
	
	// perform PCA
	myPCA(data, _mean, _eigenvectors , _eigenvalues, n);
	transpose(_eigenvectors, _eigenvectors);
		
	for(int sampleIdx = 0; sampleIdx < data.rows; sampleIdx++) {
		//投影到特征空间中，用所有特征脸分别表示每一幅图像
		Mat p = subspaceProject(_eigenvectors, _mean, data.row(sampleIdx));
		_projections.push_back(p);
	}
	
	// ----------------------------------------------------------------------------------------------------------------------//
	// -------------------------------------------------------predict-------------------------------------------------------//
	// ----------------------------------------------------------------------------------------------------------------------//
	vector<int> labels(imagesTrain.size(), 0);
	for(int i=0;i<labels.size();i+=10){
		int label = i / 10 + 1;
		for(int j=0; j<10; j++)
			labels[i+j] = label;
	}
	InputArray _local_labels = labels;
	Mat _labels = _local_labels.getMat().clone();
	ofstream paintFile("paint.dat");
	int correctNum=0, totalNum=0;
	double minDist = DBL_MAX ;
	int minClass = -1;
	int minIdx = 0;
	
	for(int i=0;i<imagesTest.size();i++){
		// The following line predicts the label of a given
		// test image:
		Mat testSample = imagesTest[i];
		imshow("1",testSample);

		//int predictedLabel = model->predict(testSample);
		// project into PCA subspace
		Mat q = subspaceProject(_eigenvectors, _mean, testSample.reshape(1,1));
				
		minDist = DBL_MAX ;
		minClass = -1;
		minIdx = 0;

		for(size_t sampleIdx = 0; sampleIdx < _projections.size(); sampleIdx++) {
			//计算测试图像跟投影空间中向量之间的欧式距离，2范数
			double dist = norm(_projections[sampleIdx], q, NORM_L2);
			if((dist < minDist) ) {
				minDist = dist;
				minClass = _labels.at<int>((int)sampleIdx);
				minIdx = (int)sampleIdx;
			}
		}

		int predictedLabel = minClass;
		int predictedImage = minIdx;

		Mat trainSample = imagesTrain[predictedImage];
		imshow("2",trainSample);
		waitKey(0);

		string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, labels[i]);
		if(i!=0)
		{
			cout<<correctNum/(1.0*totalNum)<<endl;
			paintFile<<correctNum/(1.0*totalNum);
			if(i!=imagesTest.size()-1)
				paintFile<<",";
		}
		
		if(labels[i] == predictedLabel)
			correctNum++;
		totalNum++;
	}

	waitKey();
	paintFile.close();
	return 0;
}
