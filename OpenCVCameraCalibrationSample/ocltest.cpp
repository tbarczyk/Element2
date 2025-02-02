﻿#include "stdafx.h"
#include "config.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "opencv2/ocl/ocl.hpp"
#pragma comment (lib,"OpenCL.lib")
#include <CL/cl.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
using namespace std;

cl_context context = NULL;

cl_uint ret_num_platform;
cl_device_id device_id = NULL;
cl_uint ret_num_device;
cl_command_queue command_queue = NULL;
cl_program program = NULL;
cl_kernel kernel = NULL;
cl_int err;
cl_mem image1, image2;
size_t kernel_src_size;
cl_image_format img_fmt;
cl_bool sup;
size_t rsize;
size_t origin[] = { 0, 0, 0 }; // Defines the offset in pixels in the image from where to write.
size_t region[] = { WIDTH, HEIGHT, 1 }; // Size of object to be transferred
size_t GWSize[4];
cl_event event[5];

vector<cl_int2> getElement(cv::Mat elementMatrix)
{
	vector<cl_int2> elementVector;
	for (int i = 0; i < elementMatrix.rows; i++)
	{
		for (int j = 0; j < elementMatrix.cols; j++) 
		{
			if(elementMatrix.at<unsigned char>(i,j) != 0 )
			{
				cl_int2 a = { j,i };
				elementVector.push_back(a);
			}
		}
	}
	random_shuffle(elementVector.begin(), elementVector.end());
	return elementVector;
}

void err_check(int err, string err_code) {
	if (err != CL_SUCCESS) {
  		cout << "Error: " << err_code << "(" << err << ")" << endl;
		exit(-1);
	}
}

void initOCL(cv::Mat elMat) {

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
		"for (int i = 0; i < sizeOfElement; i++) {									\n" \
		"	const int2 elementCoords = element[i];									\n" \
		"	const int2 imageCoords = coords + elementCoords - (elementDim >> 1);	\n" \
		"	const float4 imagePixel = read_imagef(image, sampler, imageCoords);		\n" \
		"	if (imagePixel.x<255) {												\n" \
		"		ans.x = 0;															\n" \
		"		ans.y = 0;															\n" \
		"		ans.z = 0;															\n" \
		"		ans.w = 0;															\n" \
		"		break;																\n" \
		"	}																		\n" \
		"}																			\n" \
		"write_imagef(imageOut, coords, ans);	};									\n";
	

	cl_uint platformIdCount = 0;
	clGetPlatformIDs(0, nullptr, &platformIdCount);
	std::vector<cl_platform_id> platform_id(platformIdCount);

	// step 1 : getting platform ID
	err = clGetPlatformIDs(platformIdCount, platform_id.data(), nullptr);
	err_check(err, "clGetPlatformIDs");
	for (int i = 0; i < platformIdCount; i++)
	{
		err = clGetDeviceIDs(platform_id[i], CL_DEVICE_TYPE_GPU, 1, &device_id, &ret_num_device);		
		if (err == CL_SUCCESS)
			break;
	}
	if (err != CL_SUCCESS)
		AfxMessageBox(CString("OpenCL device not found...app will be closed!"), MB_OK | MB_ICONEXCLAMATION);
	err_check(err, "clGetDeviceIDs");
		
	// step 2 : Get Device ID
	

	// step 3 : Create Context
	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);
	err_check(err, "clCreateContext");

	clGetDeviceInfo(device_id, CL_DEVICE_IMAGE_SUPPORT, sizeof(sup), &sup, &rsize);
	if (sup != CL_TRUE){
		cout << "Image not Supported" << endl;
	}
	// Step 4 : Create Command Queue
	command_queue = clCreateCommandQueue(context, device_id, 0, &err);
	err_check(err, "clCreateCommandQueue");

	// Step 5 : Reading Kernel Program
	kernel_src_size = sizeof(kernel_src_std);

	//  Create Image data formate
	img_fmt.image_channel_order = CL_R;
	img_fmt.image_channel_data_type = CL_FLOAT;
	
	// Step 6 : Create Image Memory Object

	image1 = clCreateImage2D(context, CL_MEM_READ_ONLY, &img_fmt, WIDTH, HEIGHT, 0, 0, &err);
	err_check(err, "image1: clCreateImage2D");

	image2 = clCreateImage2D(context, CL_MEM_READ_WRITE, &img_fmt, WIDTH, HEIGHT, 0, 0, &err);
	err_check(err, "image2: clCreateImage2D");

	int elX = elMat.cols;
	int elY = elMat.rows;
	
	vector<cl_int2> elementDataVector = getElement(elMat);
	cl_int2* elementData = &elementDataVector[0];
	cl_int2 elementDimData = {elX , elY };
	
	cl_mem element = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, elementDimData.x * elementDimData.y * 2 * sizeof(int), elementData, &err);
	err = clEnqueueWriteBuffer(command_queue, element, CL_TRUE, 0, 0, elementData, 0, NULL, &event[0]);

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
	
	int sizeOfElement = elementDataVector.size();
	
	int imageWidth = WIDTH;
	int imageHeigth = HEIGHT;

	err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&image2);
	err_check(err, "Arg 2 : clSetKernelArg");
	err = clSetKernelArg(kernel, 2, sizeof(int), (void *)&sizeOfElement);
	err = clSetKernelArg(kernel, 3, sizeof(cl_int2), (void *)&elementDimData);
	err = clSetKernelArg(kernel, 4, sizeof(int), (void *)&imageWidth);
	err = clSetKernelArg(kernel, 5, sizeof(int), (void *)&imageHeigth);
	err = clSetKernelArg(kernel, 6, sizeof(cl_mem), (void *)&element);

	GWSize[0] = WIDTH;
	GWSize[1] = HEIGHT;
	GWSize[2] = elementDimData.x * elementDimData.y * 2;
	GWSize[3] = 1;


};

cv::Mat executeKernel(cv::Mat mat_src)
{
	float c = clock();
	float cAll = clock();
	float t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	static float matData[WIDTH*HEIGHT];

	for (int i = 0; i < WIDTH*HEIGHT; i++)
		matData[i] = mat_src.data[i];

	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	
	err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&image1);
	
	err_check(err, "Arg 1 : clSetKernelArg");
	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	err = clEnqueueWriteImage(command_queue, image1, CL_TRUE, origin, region, 0, 0, matData, 0, NULL, &event[0]);
	err_check(err, "clEnqueueWriteImage");
	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	err = clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, GWSize, NULL, 1, event, &event[1]);
	static float output[WIDTH*HEIGHT];
	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	err = clEnqueueReadImage(command_queue, image2, CL_TRUE, origin, region, 0, 0, output, 2, event, &event[2]);
	
	err_check(err, "clEnqueueCopyImage");
	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	
	uchar aaa[WIDTH * HEIGHT];
	for (int i = 0; i < WIDTH * HEIGHT; i++)
		aaa[i] = (uchar)output[i];

	cv::Mat result = cv::Mat(HEIGHT, WIDTH, CV_8UC1, aaa);
	t = float(clock() - c) / CLOCKS_PER_SEC;
	c = clock();
	cv::Mat result2 = cv::Mat(HEIGHT, WIDTH, CV_8UC1);
	result.copyTo(result2);
	
	float tAll = float(clock() - cAll) / CLOCKS_PER_SEC;
	return result2;
}


