#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>

typedef int64_t i64;

static void printDeviceInfo(cl_device_id device, const char* indent)
{
	char queryBuffer[1024];
	int queryInt;
	cl_int clError;
	clError = clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(queryBuffer), &queryBuffer, 0);
	printf("%sCL_DEVICE_NAME: %s\n", indent, queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_VENDOR, sizeof(queryBuffer), &queryBuffer, 0);
	printf("%sCL_DEVICE_VENDOR: %s\n", indent, queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DRIVER_VERSION, sizeof(queryBuffer), &queryBuffer, 0);
	printf("%sCL_DRIVER_VERSION: %s\n", indent, queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_VERSION, sizeof(queryBuffer), &queryBuffer, 0);
	printf("%sCL_DEVICE_VERSION: %s\n", indent, queryBuffer);
	queryBuffer[0] = '\0';
	clError = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(int), &queryInt, 0);
	printf("%sCL_DEVICE_MAX_COMPUTE_UNITS: %d\n", indent, queryInt);
}

static void printPlatformInfo(cl_platform_id p, const char* indent)
{
	constexpr int BUFFER_SIZE = 4 * 1024;
	char buffer[BUFFER_SIZE];
	clGetPlatformInfo(p, CL_PLATFORM_NAME, BUFFER_SIZE, buffer, 0);
	printf("%sname: %s\n", indent, buffer);
	clGetPlatformInfo(p, CL_PLATFORM_VENDOR, BUFFER_SIZE, buffer, 0);
	printf("%svendor: %s\n", indent, buffer);
	clGetPlatformInfo(p, CL_PLATFORM_VERSION, BUFFER_SIZE, buffer, 0);
	printf("%sversion: %s\n", indent, buffer);
	clGetPlatformInfo(p, CL_PLATFORM_PROFILE, BUFFER_SIZE, buffer, 0);
	printf("%sprofile: %s\n", indent, buffer);
	clGetPlatformInfo(p, CL_PLATFORM_EXTENSIONS, BUFFER_SIZE, buffer, 0);
	printf("%sextensions:\n %s\n", indent, buffer);
}

static void errorCallbackCL(const char* errInfo, const void* privateInfo, size_t privateInfoSize, void* userData)
{

}

constexpr cl_uint MAX_PLATFORMS = 8;
constexpr cl_uint MAX_DEVICES = 8;

static void printPlatformsAndDevicesInfo(cl_uint numPlatforms, const cl_platform_id* platformIds)
{

	printf("numPlatforms: %d\n", numPlatforms);
	for (cl_uint i = 0; i < numPlatforms; i++) {
		printf("platform %d\n", i);
		printPlatformInfo(platformIds[i], "  ");
		cl_uint numDevices;
		cl_device_id deviceIds[MAX_DEVICES];
		clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, deviceIds, &numDevices);
		printf("  devices:\n");
		for (int j = 0; j < numDevices; j++) {
			printf("    device %d:", j);
			printDeviceInfo(deviceIds[j], "    ");
		}
		printf("\n");
	}
}

static const char* strstri(const char* container, const char* contained)
{
	const int containerLen = strlen(container);
	const int containedLen = strlen(contained);
	for (int i = 0; i + containedLen - 1 < containerLen; i++) {
		int j;
		for (j = 0; j < containedLen; j++) {
			const char a = tolower(container[i + j]);
			const char b = tolower(contained[j]);
			if (a != b)
				break;
		}
		if (i == containedLen)
			return contained + i;
	}
	return nullptr;
}

static void printErrorCode()
{

}

static void printETA(clock_t t)
{
	i64 seconds = t / CLOCKS_PER_SEC;
	i64 minutes = seconds / 60;
	seconds = seconds % 60;
	printf("ETA: %" PRId64 " minutes, %" PRId64 " seconds\n", minutes, seconds);
}

constexpr i64 NANO = 1'000'000'000;
constexpr i64 NT = 60 * 60 * 12;
constexpr i64 N = NANO * NT;
constexpr i64 perThread = 100'000; // how many iterations we will do in each work-item
constexpr i64 numThreads = 10'000;
constexpr i64 perIteration = perThread * numThreads;
//constexpr i64 iterations = N / perIteration;

const char* kernelCode =
R"CL(
#define NANO ((long)1000000000)
__kernel void search(__global long* out, long offset, long perThread)
{
	size_t id = get_global_id(0);
	out[id] = 0;
	long startRange = offset + id * perThread;
	long endRange = startRange + perThread;
	for(long i = startRange; i <  endRange; i++) {
		if((11 * i) % (NANO * 60*60*12) == 1) {
			out[id] = i;
			break;
		}
	}
}
)CL";

int main()
{
	cl_uint numPlatforms;
	cl_platform_id platformIds[MAX_PLATFORMS];
	clGetPlatformIDs(MAX_PLATFORMS, platformIds, &numPlatforms);

	printPlatformsAndDevicesInfo(numPlatforms, platformIds);

	cl_platform_id bestGpuPlatform = nullptr;
	cl_device_id bestGpuDevice = nullptr;
	for (int platformInd = 0; platformInd < numPlatforms; platformInd++) {
		cl_uint numGpuDevices = 0;
		cl_device_id gpuDevices[MAX_DEVICES];
		clGetDeviceIDs(platformIds[platformInd], CL_DEVICE_TYPE_GPU, MAX_DEVICES, gpuDevices, &numGpuDevices);
		for (int i = 0; i < numGpuDevices; i++) {
			constexpr int BUFFER_SIZE = 256;
			char buffer[BUFFER_SIZE];
			clGetDeviceInfo(gpuDevices[i], CL_DEVICE_NAME, BUFFER_SIZE, &buffer, 0);
			if (bestGpuDevice == nullptr || strstri(buffer, "nvidia")) {
				bestGpuDevice = gpuDevices[i];
				bestGpuPlatform = platformIds[platformInd];
			}
		}
	}

	const cl_context_properties ctxProps[] = {CL_CONTEXT_PLATFORM, cl_context_properties(bestGpuPlatform), 0};
	cl_int errorCode;
	cl_build_status status;
	const cl_context clCtx = clCreateContext(ctxProps, 1, &bestGpuDevice, errorCallbackCL, nullptr, &errorCode);

	const cl_command_queue cmdQueue = clCreateCommandQueue(clCtx, bestGpuDevice, 0, &errorCode);
	i64* result = new i64[perIteration];
	//memset(result, 0, sizeof(i64) * perIteration);
	const cl_mem resultBuffer = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, sizeof(i64) * numThreads, nullptr, &errorCode);
	const cl_program prog = clCreateProgramWithSource(clCtx, 1, &kernelCode, nullptr, &errorCode);
	if (errorCode != 0)
		printf("error creating kernel\n");
	status = clBuildProgram(prog, 1, &bestGpuDevice, 0, 0, 0);
	if (status != CL_SUCCESS) {
		if (status == CL_BUILD_PROGRAM_FAILURE) {
			printf("Error building kernel\n");
			char buffer[4 * 1024];
			size_t len;
			clGetProgramBuildInfo(prog, bestGpuDevice, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
			printf("%s\n", buffer);
		}
		else
			printf("Unrecognized error building kernel\n");
	}
	const cl_kernel kernel = clCreateKernel(prog, "search", &status);
	if (status != CL_SUCCESS)
		printf("Error getting kernel\n");
	clSetKernelArg(kernel, 0, sizeof(cl_mem), &resultBuffer);
	clSetKernelArg(kernel, 2, sizeof(cl_long), &perThread);

	const clock_t startT = clock();

	cl_long percent = 0;
	for (cl_long i = 0; i < N / perIteration; i++)
	{
		cl_long offset = i * perIteration;
		cl_long newPercent = (100 * offset) / N;
		if (newPercent > percent) {
			printf("%" PRId64 "%%\n", newPercent);
			const clock_t elapsedT = clock() - startT;
			const clock_t eta = elapsedT * (100 - newPercent) / newPercent;
			printETA(eta);
			percent = newPercent;
		}
		clSetKernelArg(kernel, 1, sizeof(cl_long), &offset);
		static const size_t spaceSize[1] = {numThreads};
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, spaceSize, nullptr, 0, nullptr, nullptr);
		clEnqueueReadBuffer(cmdQueue, resultBuffer, CL_TRUE, 0, sizeof(i64) * numThreads, result, 0, nullptr, nullptr);
		bool found = false;
		for (int i = 0; i < numThreads; i++) {
			if (result[i]) {
				printf("RESULT: %" PRId64 "\n", result[i]);
				found = true;
				break;
			}
		}
		if (found)
			break;
	}

}