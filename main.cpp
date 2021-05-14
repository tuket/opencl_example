#include <stdio.h>
#include <string.h>
#include <ctype.h>
#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <assert.h>
#include <vulkan/vulkan.h>

typedef int32_t i32;
typedef uint32_t u32;
typedef int64_t i64;
typedef uint64_t u64;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

constexpr i64 NANO = 1'000'000'000;
constexpr i64 NT = 60 * 60 * 12;
constexpr i64 N = NANO * NT;
constexpr i64 perThread = 1'000'000; // how many iterations we will do in each work-item
constexpr i64 numThreads = 10'000;
constexpr i64 perIteration = perThread * numThreads;
constexpr i64 numIterations = N / perIteration;

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
		if (j == containedLen)
			return contained + i;
	}
	return nullptr;
}

static double getTime()
{
	timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return t.tv_sec + 1e-9*t.tv_nsec;
}

static void printETA(double t)
{
	i64 seconds = t;
	i64 minutes = seconds / 60;
	i64 hours = minutes / 60;
	minutes %= 60;
	seconds %= 60;
	printf("ETA: %" PRId64 "hours, %" PRId64 " minutes, %" PRId64 " seconds\n", hours, minutes, seconds);
}


static void calcElapsedAndPrintETA(double startT, i64 done, i64 total)
{
	const double nowT = getTime();
	const double elapsedT = nowT - startT;
	const double eta = elapsedT * (total - done) / done;
	printETA(eta);
}

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

static const char* errorToStr(cl_int err)
{
	switch(err)
	{
		case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
		case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
		case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
		case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
	}
	return "[?]";
}

// --- CPU implementation single threaded ---
static i64 calcWithCpu()
{
	constexpr i64 batch = 1'000'000'000;
	const double startT = getTime();
	i64 percent = 0;
	for (i64 i = 0; i < N;) {
		const i64 newPercent = (i64(10000) * i) / N;
		if (newPercent != percent) {
			printf("%" PRId64 ".%" PRId64 "%%\n", newPercent / 100, newPercent % 100);
			percent = newPercent;
			
		}
		if (i != 0) {
			calcElapsedAndPrintETA(startT, i, N);
		}

		i64 n = MIN(i + batch, N);
		for(; i < n; i++)
		if ((i64(11) * i) % N == 1)
			return i;
	}
	assert(false);
	return 0;
}

// --- OpenCL implementation ---

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

i64 calcWithOpenCl()
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
	i64* result = new i64[numThreads];
	//memset(result, 0, sizeof(i64) * perIteration);
	const cl_mem resultBuffer = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, sizeof(i64) * numThreads, nullptr, &errorCode);
	const cl_program prog = clCreateProgramWithSource(clCtx, 1, &kernelCode, nullptr, &errorCode);
	if (errorCode != 0) {
		printf("error creating kernel\n");
		printf("%s\n", errorToStr(errorCode));
	}
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

	const double startT = getTime();

	cl_long percent = 0;
	for (cl_long i = 0; i < numIterations; i++)
	{
		cl_long offset = i * perIteration;
		cl_long newPercent = (10000 * offset) / N;
		if (newPercent > percent) {
			printf("%" PRId64 ".%" PRId64 "%%\n", newPercent / 100, newPercent % 100);
			percent = newPercent;
			calcElapsedAndPrintETA(startT, i, numIterations);
		}
		clSetKernelArg(kernel, 1, sizeof(cl_long), &offset);
		static const size_t spaceSize[1] = {numThreads};
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, spaceSize, nullptr, 0, nullptr, nullptr);
		clEnqueueReadBuffer(cmdQueue, resultBuffer, CL_TRUE, 0, sizeof(i64) * numThreads, result, 0, nullptr, nullptr);
		for (int i = 0; i < numThreads; i++) {
			if (result[i])
				return result[i];
		}
	}

	assert(false);
	return 0;
}

// --- Vulkan implementation ---
static void printPhysicalDeviceProps(VkPhysicalDeviceProperties props)
{
	printf("device name: %s\n", props.deviceName);
}
static void printPhysicalDeviceProps(VkPhysicalDevice gpu)
{
	VkPhysicalDeviceProperties props;
	vkGetPhysicalDeviceProperties(gpu, &props);
	printPhysicalDeviceProps(props);
}

static void printMemoryProps(const VkPhysicalDeviceMemoryProperties& memProps)
{
	printf("Memory Types:\n");
	for(int i = 0; i < memProps.memoryTypeCount; i++) {
		auto& t = memProps.memoryTypes[i];
		printf("%d)\n", i);
		printf("  Heap index: %d\n", t.heapIndex);
		printf("  Flags:");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			printf(" VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_CACHED_BIT |");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
			printf(" VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |");
		if(t.propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
			printf(" VK_MEMORY_PROPERTY_PROTECTED_BIT |");
			
		printf("\n");
	}

	printf("\nMemory Heaps:\n");
	for(int i = 0; i < memProps.memoryHeapCount; i++) {
		auto& heap = memProps.memoryHeaps[i];
		printf("%d)\n", i);
		printf("  Size: %" PRIu64 "MB\n", heap.size / (1024 * 1024));
		printf("  Flags:");
		if(heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			printf(" VK_MEMORY_HEAP_DEVICE_LOCAL_BIT");
		printf("\n");
	}
}

static i64 calcWithVulkan()
{
	constexpr u32 MAX_LAYERS = 32;
	VkLayerProperties layersProps[MAX_LAYERS];
	u32 numLayers = MAX_LAYERS;
	vkEnumerateInstanceLayerProperties(&numLayers, layersProps);
	printf("Available Layers\n");
	for(u32 i = 0; i < numLayers; i++) {
		printf("%s: %s\n", layersProps[i].layerName, layersProps[i].description);
	}

	VkApplicationInfo appInfo = {};
	appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
	appInfo.pApplicationName = "example";
	appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.pEngineName = "none";
	appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
	appInfo.apiVersion = VK_API_VERSION_1_0;

	VkInstanceCreateInfo instanceCreateInfo = {};
	instanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
	instanceCreateInfo.pApplicationInfo = &appInfo;
	instanceCreateInfo.enabledLayerCount = 1;
	static const char* const LAYER_NAMES[] = {"VK_LAYER_KHRONOS_validation"};
	instanceCreateInfo.ppEnabledLayerNames = LAYER_NAMES;
	VkInstance inst;
	VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &inst);

	if(result != VK_SUCCESS) {
		printf("Error creating Vulkan instance\n");
		return 0;
	}

	constexpr u32 MAX_PHYSICAL_DEVICES = 4;
	VkPhysicalDevice physicalDevices[MAX_PHYSICAL_DEVICES];
	u32 numPhysicalDevices = MAX_PHYSICAL_DEVICES;
	vkEnumeratePhysicalDevices(inst, &numPhysicalDevices, physicalDevices);
	assert(numPhysicalDevices > 0);

	VkPhysicalDevice bestGpu = nullptr;
	for(u32 i = 0; i < numPhysicalDevices; i++) {
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
		//printPhyiscalDeviceProps(props);
		if(!bestGpu || strstri(props.deviceName, "geforce"))
			bestGpu = physicalDevices[i];
	}
	printf("best GPU: ");
	printPhysicalDeviceProps(bestGpu);

	constexpr u32 MAX_QUEUE_FAMILIES = 8;
	VkQueueFamilyProperties queueFamiles[MAX_QUEUE_FAMILIES];
	u32 numQueueFamilies = MAX_QUEUE_FAMILIES;
	vkGetPhysicalDeviceQueueFamilyProperties(bestGpu, &numQueueFamilies, queueFamiles);
	printf("numQueueFamilies: %d\n", numQueueFamilies);
	for(int i = 0; i < numQueueFamilies; i++) {
		printf("%d)\n", i);
		printf("flags: ");
		if(queueFamiles[i].queueFlags | VK_QUEUE_COMPUTE_BIT)
			printf("COMPUTE | ");
		if(queueFamiles[i].queueFlags | VK_QUEUE_GRAPHICS_BIT)
			printf("GRAPHICS | ");
		if(queueFamiles[i].queueFlags | VK_QUEUE_TRANSFER_BIT)
			printf("TRANSFER | ");
		if(queueFamiles[i].queueFlags | VK_QUEUE_SPARSE_BINDING_BIT)
			printf("SPARSE_BINDING | ");
			
		printf("\n");
		printf("queueCount: %d\n", queueFamiles[i].queueCount);
	}
	u32 bestQueueFamilyInd = -1;
	for(u32 i = 0; i < numQueueFamilies; i++) {
		if( (queueFamiles[i].queueFlags | VK_QUEUE_COMPUTE_BIT) &&
			(
				bestQueueFamilyInd == -1 ||
				queueFamiles[i].queueCount > queueFamiles[bestQueueFamilyInd].queueCount
			)
		) {
			bestQueueFamilyInd = i;
		}
	}

	const float queuePriorities[] = {0};
	VkDeviceQueueCreateInfo queueCreateInfo = {};
	queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	queueCreateInfo.queueCount = 1;
	queueCreateInfo.queueFamilyIndex = bestQueueFamilyInd;
	queueCreateInfo.pQueuePriorities = queuePriorities;

	VkPhysicalDeviceFeatures gpuFeatures;
	vkGetPhysicalDeviceFeatures(bestGpu, &gpuFeatures);
	if(!gpuFeatures.shaderInt64) {
		printf("Error: 64 bit integer not supported\n");
		return 0;
	}
	memset(&gpuFeatures, 0, sizeof(gpuFeatures));
	gpuFeatures.shaderInt64 = VK_TRUE;
	
	VkDeviceCreateInfo deviceCreateInfo = {};
	deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
	deviceCreateInfo.queueCreateInfoCount = 1;
	deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
	deviceCreateInfo.pEnabledFeatures = &gpuFeatures;
	VkDevice device;
	vkCreateDevice(bestGpu, &deviceCreateInfo, nullptr, &device);

	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(bestGpu, &memProps);
	printMemoryProps(memProps);

	VkBufferCreateInfo bufferCreateInfo = {};
	bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	bufferCreateInfo.size = sizeof(i64) * numThreads;
	bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
	bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	VkBuffer buffer;
	vkCreateBuffer(device, &bufferCreateInfo, nullptr, &buffer);

	VkMemoryRequirements bufferMemReqs;
	vkGetBufferMemoryRequirements(device, buffer, &bufferMemReqs);

	VkDeviceMemory bufferMem = nullptr;
	//VkMemoryPropertyFlagBits memPropFlags;
	VkMemoryAllocateInfo allocInfo;
	allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	allocInfo.allocationSize = bufferMemReqs.size;
	for(u32 i = 0; i < memProps.memoryTypeCount; i++) {
		const VkMemoryPropertyFlags typeFlags = memProps.memoryTypes[i].propertyFlags;
		if(typeFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
			allocInfo.memoryTypeIndex = i;
			break;
		}
	}
	vkAllocateMemory(device, &allocInfo, nullptr, &bufferMem);
	assert(bufferMem && "Error allocating memory");
	vkBindBufferMemory(device, buffer, bufferMem, 0);




	return 0;
}

int main()
{
	i64 res =
		//calcWithCpu();
		//calcWithOpenCl();
		calcWithVulkan();
	printf("RESULT: %" PRId64 "\n", res);
}