#include "stdafx.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#define IMG_SIZE 512
#define WIDTH 512
#define HEIGHT 512
#define IMAGE_NAME "lena3.bmp"
//#include <boost/compute.hpp>
//#include <boost/compute/interop/opencv/core.hpp>
#include "opencv2/ocl/ocl.hpp"
#pragma comment (lib,"OpenCL.lib")
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
using namespace std;

vector<cl_int2> getElement(int x, int y)
{
	cv::Mat elementMatrix = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(x, y));
	vector<cl_int2> elementVector;
	unsigned char * matrix = (unsigned char*)(elementMatrix.data);
	for (int i = 0; i < elementMatrix.cols; i++)
	{
		for (int j = 0; j < elementMatrix.rows; j++)
		{
			if (matrix[elementMatrix.cols*j + i] != 0)
			{
				cl_int2 a = { i,j };
				elementVector.push_back(a);
			}
		}

	}
	
	return elementVector;
}


void err_check(int err, string err_code) {
	if (err != CL_SUCCESS) {
		cout << "Error: " << err_code << "(" << err << ")" << endl;
		exit(-1);
	}
}

cv::Mat testOCL2() {

	char* kernel_src_std =
		"__kernel void image_copy(read_only image2d_t image, write_only image2d_t imageOut, int sizeOfElement, int2 elementDim, int imageWidth, int imageHeight, __global read_only int2* element)						 \n" \
		"{																											 \n" \
		"const int2 coords = { get_global_id(0), get_global_id(1) };				\n" \
		"if (coords.x >= imageWidth || coords.y >= imageHeight){					\n" \
		"	return;																	\n" \
		"}																			\n" \
		"const sampler_t sampler =													\n" \
		"	CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;	\n" \
		"float4 ans = { 255, 255, 255, 255 };										\n" \
		"for (int i = 0; i < sizeOfElement; ++i) {									\n" \
		"	const int2 elementCoords = element[i];									\n" \
		"	const int2 imageCoords = coords + elementCoords - (elementDim >> 1);	\n" \
		"	const float4 imagePixel = read_imagef(image, sampler, imageCoords);		\n" \
		"	if (imagePixel.x < 255) {												\n" \
		"		ans.x = 0;															\n" \
		"		ans.y = 0;															\n" \
		"		ans.z = 0;															\n" \
		"		ans.w = 0;															\n" \
		"		break;																\n" \
		"	}																		\n" \
		"}																			\n" \
		"write_imagef(imageOut, coords, ans);	};									\n";
		/*"																											 \n" \
		"	const int xout = get_global_id(0);																		 \n" \
		"	const int yout = get_global_id(1);																		 \n" \
		"	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;			 \n" \
		"	float4 pixel;																							 \n" \

		"	pixel = read_imagef(image, sampler, (int2)(xout, yout));												 \n" \
		"	pixel.x=pixel.x/(elementDim.y);																								 \n" \
		"	pixel.y=pixel.y/(elementDim.y);																										 \n" \
		"	pixel.z=pixel.z/(elementDim.y);																											 \n" \
		"	pixel.w=pixel.w/(elementDim.y);																									 \n" \
		"	write_imagef(imageOut, (int2)(xout, yout), pixel);														 \n" \
//		"}		*///																									 \n";
	
	//zauważ ze jest element[2].x*10 w kodzie kernela zeby sprawdzic czy element sie dobrze przekazuje w postaci int2 * wtedy obraz ma wartosc w zaleznosci od podanego wartosci elementu
	//normalnie mozna dac wartosc obrazka wejsciowego np. pixel.x = pixel.x / 10  << to przyciemnia obraz bo pomniejsza wartosc piksela danego
	
	cl_platform_id platform_id = NULL;
	cl_uint ret_num_platform;

	cl_device_id device_id = NULL;
	cl_uint ret_num_device;

	cl_context context = NULL;

	cl_command_queue command_queue = NULL;

	cl_program program = NULL;

	cl_kernel kernel = NULL;

	cl_int err;

	//float input[IMG_SIZE * 3];

	static float output[WIDTH*HEIGHT];

	//// Create Input data
	//for (int i = 0; i<3; i++){
	//	for (int j = 0; j<IMG_SIZE; ++j){

	//		input[(i*IMG_SIZE) + j] = (float)(j + 1);

	//	}

	//}
	cv::Mat mat_src = cv::imread(IMAGE_NAME, cv::IMREAD_GRAYSCALE);
	static float matData[WIDTH*HEIGHT];
	
	for (int i = 0; i < mat_src.rows; i++)
	{
		for (int j = 0; j < mat_src.cols; j++)
			matData[i*512+j] = mat_src.at<unsigned char>(i, j);
	}

	// step 1 : getting platform ID
	err = clGetPlatformIDs(1, &platform_id, &ret_num_platform);
	err_check(err, "clGetPlatformIDs");

	// step 2 : Get Device ID
	err = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_device);
	err_check(err, "clGetDeviceIDs");

	// step 3 : Create Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	err_check(err, "clCreateContext");

	cl_bool sup;
	size_t rsize;
	clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(sup), &sup, &rsize);
	if (sup != CL_TRUE){
		cout << "Image not Supported" << endl;
	}
	// Step 4 : Create Command Queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	err_check(err, "clCreateCommandQueue");

	// Step 5 : Reading Kernel Program

	
	size_t kernel_src_size;


	kernel_src_size = sizeof(kernel_src_std);

	//  Create Image data formate
	cl_image_format img_fmt;

	img_fmt.image_channel_order = CL_R;
	img_fmt.image_channel_data_type = CL_FLOAT;

	// Step 6 : Create Image Memory Object
	cl_mem image1, image2;

	size_t width, height;
	//width = height = 512;// sqrt(IMG_SIZE);

	image1 = clCreateImage2D(context, CL_MEM_READ_ONLY, &img_fmt, WIDTH, HEIGHT, 0, 0, &err);
	err_check(err, "image1: clCreateImage2D");

	image2 = clCreateImage2D(context, CL_MEM_READ_WRITE, &img_fmt, WIDTH, HEIGHT, 0, 0, &err);
	err_check(err, "image2: clCreateImage2D");

	// Copy Data from Host to Device
	cl_event event[5];

	size_t origin[] = { 0, 0, 0 }; // Defines the offset in pixels in the image from where to write.
	size_t region[] = { WIDTH, HEIGHT, 1 }; // Size of object to be transferred
	err = clEnqueueWriteImage(command_queue, image1, CL_TRUE, origin, region, 0, 0, matData, 0, NULL, &event[0]);
	err_check(err, "clEnqueueWriteImage");
	
	//cl_int2 elementData[] = { { 11, 12 }, { 21, 22 }, { 31, 32 }, { 41, 42 } };

	//TODO: to nie moze byc tak, musi byc max X - min X i max Y - min Y zamiast na sztywno wstawionych wymiarow boxa elementu
	int elX = 10;
	int elY = 20;


	vector<cl_int2> elementDataVector = getElement(elX,elY);
	cl_int2* elementData = &elementDataVector[0];
	cl_int2 elementDimData = { elX, elY };
	//int elementData[elX * elY * 2] = { 1, 2, 3, 4, 5, 6, 7, 20 };

	cl_mem element = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, elementDimData.x * elementDimData.y * 2 * sizeof(int), elementData, &err);
	err = clEnqueueWriteBuffer(command_queue, element, CL_TRUE, 0, 0, elementData, 0, NULL, &event[0]);



	/*cl_mem elementDim = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 2 * sizeof(int), &elementDimData, &err);
	err = clEnqueueWriteBuffer(command_queue, elementDim, CL_TRUE, 0, 0, &elementDimData, 0, NULL, &event[0]);
	
	*///cout<<kernel_src_std;
	// Step 7 : Create and Build Program
	program = clCreateProgramWithSource(context, 1, (const char **)&kernel_src_std, 0, &err);
	err_check(err, "clCreateProgramWithSource");

	err = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

	if (err == CL_BUILD_PROGRAM_FAILURE)
		cout << "clBulidProgram Fail...." << endl;
	err_check(err, "clBuildProgram");

	// Step 8 : Create Kernel
	kernel = clCreateKernel(program, "image_copy", &err);

	// Step 9 : Set Kernel Arguments

	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image1);
	err_check(err, "Arg 1 : clSetKernelArg");

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image2);
	err_check(err, "Arg 2 : clSetKernelArg");
	
	int sizeOfElement = elementDataVector.size();
	
	int imageWidth = 512;
	int imageHeigth = 512;

	err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&sizeOfElement);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int2), (void *)&elementDimData);
	err = clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageWidth);
	err = clSetKernelArg(kernel, 5, sizeof(int), (void *)&imageHeigth);
	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&element);
	//err = clSetKernelArg(kernel, 8, sizeof(cl_mem), (void *)&elementDim);

	// Step 10 : Execute Kernel 
	size_t GWSize[] = { WIDTH, HEIGHT, elementDimData.x * elementDimData.y * 2, 1 };

	//czemu razy dwa???????   /\/\/\/\/\/\


	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, GWSize, NULL, 1, event, &event[1]);

	// Step 11 : Read output Data, from Device to Host
	err = clEnqueueReadImage(command_queue, image2, CL_TRUE, origin, region, 0, 0, output, 2, event, &event[2]);

	// Print Output
	float a = 0;
	for (int i = 0; i<WIDTH; i++){
		for (int j = 0; j<HEIGHT; ++j){

			a = output[(i*WIDTH) + j];

		}

	}

	cl_mem image3;

	image3 = clCreateImage2D(context, CL_MEM_READ_WRITE, &img_fmt, WIDTH, HEIGHT, 0, 0, &err);

	// copy Image1 to Image3
	err = clEnqueueCopyImage(command_queue, image1, image3, origin, origin, region, 1, event, &event[3]);
	err_check(err, "clEnqueueCopyImage");
	uchar aaa[512 * 512];
	for (int i = 0; i < 512 * 512; i++)
		aaa[i] = (uchar)output[i];
	

	cv::Mat result = cv::Mat(512, 512, CV_8UC1, aaa);
	cv::namedWindow("res");
	cv::imshow("res", result);
	
	// Step 12 : Release Objects
	//cv::waitKey(0);
	clReleaseMemObject(image3);
	clReleaseMemObject(image1);
	clReleaseMemObject(image2);
	clReleaseKernel(kernel);
	clReleaseProgram(program);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return result;
	//free(kernel_src_std);

	
};


