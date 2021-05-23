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

static const char *strstri(const char *container, const char *contained)
{
	const int containerLen = strlen(container);
	const int containedLen = strlen(contained);
	for (int i = 0; i + containedLen - 1 < containerLen; i++)
	{
		int j;
		for (j = 0; j < containedLen; j++)
		{
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

#ifdef _WIN32
#include <windows.h>
#endif

static double getTime()
{
#ifdef _WIN32
	LARGE_INTEGER t, f;
	bool success = QueryPerformanceCounter(&t);
	success &= QueryPerformanceFrequency(&f);
	assert(success);
	return double(t.QuadPart) / double(f.QuadPart);
#else
	timespec t;
	clock_gettime(CLOCK_REALTIME, &t);
	return t.tv_sec + 1e-9 * t.tv_nsec;
#endif
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

static void printDeviceInfo(cl_device_id device, const char *indent)
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

static void printPlatformInfo(cl_platform_id p, const char *indent)
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

static void errorCallbackCL(const char *errInfo, const void *privateInfo, size_t privateInfoSize, void *userData)
{
}

constexpr cl_uint MAX_PLATFORMS = 8;
constexpr cl_uint MAX_DEVICES = 8;

static void printPlatformsAndDevicesInfo(cl_uint numPlatforms, const cl_platform_id *platformIds)
{

	printf("numPlatforms: %d\n", numPlatforms);
	for (cl_uint i = 0; i < numPlatforms; i++)
	{
		printf("platform %d\n", i);
		printPlatformInfo(platformIds[i], "  ");
		cl_uint numDevices;
		cl_device_id deviceIds[MAX_DEVICES];
		clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, MAX_DEVICES, deviceIds, &numDevices);
		printf("  devices:\n");
		for (int j = 0; j < numDevices; j++)
		{
			printf("    device %d:", j);
			printDeviceInfo(deviceIds[j], "    ");
		}
		printf("\n");
	}
}

static const char *errorToStr(cl_int err)
{
	switch (err)
	{
	case CL_INVALID_CONTEXT:
		return "CL_INVALID_CONTEXT";
	case CL_INVALID_VALUE:
		return "CL_INVALID_VALUE";
	case CL_OUT_OF_RESOURCES:
		return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY:
		return "CL_OUT_OF_HOST_MEMORY";
	}
	return "[?]";
}

// --- CPU implementation single threaded ---
static i64 calcWithCpu()
{
	constexpr i64 batch = 1'000'000'000;
	const double startT = getTime();
	i64 percent = 0;
	for (i64 i = 0; i < N;)
	{
		const i64 newPercent = (i64(10000) * i) / N;
		if (newPercent != percent)
		{
			printf("%" PRId64 ".%" PRId64 "%%\n", newPercent / 100, newPercent % 100);
			percent = newPercent;
		}
		if (i != 0)
		{
			calcElapsedAndPrintETA(startT, i, N);
		}

		i64 n = MIN(i + batch, N);
		for (; i < n; i++)
			if ((i64(11) * i) % N == 1)
				return i;
	}
	assert(false);
	return 0;
}

// --- OpenCL implementation ---

const char *kernelCode =
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
	for (int platformInd = 0; platformInd < numPlatforms; platformInd++)
	{
		cl_uint numGpuDevices = 0;
		cl_device_id gpuDevices[MAX_DEVICES];
		clGetDeviceIDs(platformIds[platformInd], CL_DEVICE_TYPE_GPU, MAX_DEVICES, gpuDevices, &numGpuDevices);
		for (int i = 0; i < numGpuDevices; i++)
		{
			constexpr int BUFFER_SIZE = 256;
			char buffer[BUFFER_SIZE];
			clGetDeviceInfo(gpuDevices[i], CL_DEVICE_NAME, BUFFER_SIZE, &buffer, 0);
			if (bestGpuDevice == nullptr || strstri(buffer, "nvidia"))
			{
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
	i64 *result = new i64[numThreads];
	//memset(result, 0, sizeof(i64) * perIteration);
	const cl_mem resultBuffer = clCreateBuffer(clCtx, CL_MEM_WRITE_ONLY, sizeof(i64) * numThreads, nullptr, &errorCode);
	const cl_program prog = clCreateProgramWithSource(clCtx, 1, &kernelCode, nullptr, &errorCode);
	if (errorCode != 0)
	{
		printf("error creating kernel\n");
		printf("%s\n", errorToStr(errorCode));
	}
	status = clBuildProgram(prog, 1, &bestGpuDevice, 0, 0, 0);
	if (status != CL_SUCCESS)
	{
		if (status == CL_BUILD_PROGRAM_FAILURE)
		{
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
		if (newPercent > percent)
		{
			printf("%" PRId64 ".%" PRId64 "%%\n", newPercent / 100, newPercent % 100);
			percent = newPercent;
			calcElapsedAndPrintETA(startT, i, numIterations);
		}
		clSetKernelArg(kernel, 1, sizeof(cl_long), &offset);
		static const size_t spaceSize[1] = {numThreads};
		status = clEnqueueNDRangeKernel(cmdQueue, kernel, 1, nullptr, spaceSize, nullptr, 0, nullptr, nullptr);
		clEnqueueReadBuffer(cmdQueue, resultBuffer, CL_TRUE, 0, sizeof(i64) * numThreads, result, 0, nullptr, nullptr);
		for (int i = 0; i < numThreads; i++)
		{
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

static void printMemoryProps(const VkPhysicalDeviceMemoryProperties &memProps)
{
	printf("Memory Types:\n");
	for (int i = 0; i < memProps.memoryTypeCount; i++)
	{
		auto &t = memProps.memoryTypes[i];
		printf("%d)\n", i);
		printf("  Heap index: %d\n", t.heapIndex);
		printf("  Flags:");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
			printf(" VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)
			printf(" VK_MEMORY_PROPERTY_HOST_CACHED_BIT |");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)
			printf(" VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT |");
		if (t.propertyFlags & VK_MEMORY_PROPERTY_PROTECTED_BIT)
			printf(" VK_MEMORY_PROPERTY_PROTECTED_BIT |");

		printf("\n");
	}

	printf("\nMemory Heaps:\n");
	for (int i = 0; i < memProps.memoryHeapCount; i++)
	{
		auto &heap = memProps.memoryHeaps[i];
		printf("%d)\n", i);
		printf("  Size: %" PRIu64 "MB\n", heap.size / (1024 * 1024));
		printf("  Flags:");
		if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT)
			printf(" VK_MEMORY_HEAP_DEVICE_LOCAL_BIT");
		printf("\n");
	}
}

inline size_t aligned(size_t offset, size_t alignment)
{
	offset = (offset + alignment - 1) / alignment * alignment;
	return offset;
}

static i64 calcWithVulkan()
{
	constexpr u32 MAX_LAYERS = 32;
	VkLayerProperties* layersProps = new VkLayerProperties[MAX_LAYERS];
	// TODO: delete[]
	u32 numLayers = MAX_LAYERS;
	vkEnumerateInstanceLayerProperties(&numLayers, layersProps);
	printf("Available Layers\n");
	for (u32 i = 0; i < numLayers; i++)
	{
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
	instanceCreateInfo.enabledExtensionCount = 1;
	static const char* const EXTENSION_NAMES[] = {"VK_EXT_debug_utils"};
	instanceCreateInfo.ppEnabledExtensionNames = EXTENSION_NAMES;
	instanceCreateInfo.enabledLayerCount = 1;
	static const char *const LAYER_NAMES[] = {"VK_LAYER_KHRONOS_validation"};
	instanceCreateInfo.ppEnabledLayerNames = LAYER_NAMES;
	VkInstance inst;
	VkResult result = vkCreateInstance(&instanceCreateInfo, nullptr, &inst);

	if (result != VK_SUCCESS)
	{
		printf("Error creating Vulkan instance\n");
		return 0;
	}

	constexpr u32 MAX_PHYSICAL_DEVICES = 4;
	VkPhysicalDevice* physicalDevices = new VkPhysicalDevice[MAX_PHYSICAL_DEVICES];
	// TODO: delete[]
	u32 numPhysicalDevices = MAX_PHYSICAL_DEVICES;
	vkEnumeratePhysicalDevices(inst, &numPhysicalDevices, physicalDevices);
	assert(numPhysicalDevices > 0);

	VkPhysicalDevice bestGpu = nullptr;
	for (u32 i = 0; i < numPhysicalDevices; i++)
	{
		VkPhysicalDeviceProperties props;
		vkGetPhysicalDeviceProperties(physicalDevices[i], &props);
		//printPhyiscalDeviceProps(props);
		if (!bestGpu || strstri(props.deviceName, "geforce"))
			bestGpu = physicalDevices[i];
	}
	printf("best GPU: ");
	printPhysicalDeviceProps(bestGpu);

	constexpr u32 MAX_QUEUE_FAMILIES = 8;
	VkQueueFamilyProperties* queueFamiles = new VkQueueFamilyProperties[MAX_QUEUE_FAMILIES];
	// TODO: delete[]
	u32 numQueueFamilies = MAX_QUEUE_FAMILIES;
	vkGetPhysicalDeviceQueueFamilyProperties(bestGpu, &numQueueFamilies, queueFamiles);
	printf("numQueueFamilies: %d\n", numQueueFamilies);
	for (int i = 0; i < numQueueFamilies; i++)
	{
		printf("%d)\n", i);
		printf("flags: ");
		if (queueFamiles[i].queueFlags | VK_QUEUE_COMPUTE_BIT)
			printf("COMPUTE | ");
		if (queueFamiles[i].queueFlags | VK_QUEUE_GRAPHICS_BIT)
			printf("GRAPHICS | ");
		if (queueFamiles[i].queueFlags | VK_QUEUE_TRANSFER_BIT)
			printf("TRANSFER | ");
		if (queueFamiles[i].queueFlags | VK_QUEUE_SPARSE_BINDING_BIT)
			printf("SPARSE_BINDING | ");

		printf("\n");
		printf("queueCount: %d\n", queueFamiles[i].queueCount);
	}
	u32 bestQueueFamilyInd = -1;
	for (u32 i = 0; i < numQueueFamilies; i++)
	{
		if ((queueFamiles[i].queueFlags | VK_QUEUE_COMPUTE_BIT) &&
			(bestQueueFamilyInd == -1 ||
			 queueFamiles[i].queueCount > queueFamiles[bestQueueFamilyInd].queueCount))
		{
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
	if (!gpuFeatures.shaderInt64)
	{
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

	VkQueue queue;
	vkGetDeviceQueue(device, bestQueueFamilyInd, 0, &queue);

	VkPhysicalDeviceMemoryProperties memProps;
	vkGetPhysicalDeviceMemoryProperties(bestGpu, &memProps);
	printMemoryProps(memProps);
	const u32 localMemTypeInd = [&memProps]() -> u32 {
		for (u32 i = 0; i < memProps.memoryTypeCount; i++)
		{
			const VkMemoryPropertyFlags typeFlags = memProps.memoryTypes[i].propertyFlags;
			if (typeFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
				return i;
		}
		assert(false);
	}();
	const u32 hostVisibleMemTypeInd = [&memProps]() -> u32 {
		for (u32 i = 0; i < memProps.memoryTypeCount; i++)
		{
			const VkMemoryPropertyFlags typeFlags = memProps.memoryTypes[i].propertyFlags;
			if (typeFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)
				return i;
		}
		assert(false);
	}();

	struct Uniforms { i64 start, perThread, N; };
	Uniforms uniforms = { 0, perThread, N };

	VkBuffer buffer;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(i64) * numThreads;
		info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &info, nullptr, &buffer);
	}

	VkBuffer unifsBuffer;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(Uniforms);
		info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &info, nullptr, &unifsBuffer);
	}

	VkBuffer stagingBuffer;
	{
		VkBufferCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
		info.size = sizeof(i64) * numThreads;
		info.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
		info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
		vkCreateBuffer(device, &info, nullptr, &stagingBuffer);
	}

	VkDeviceMemory bufferMem = nullptr;
	{
		VkMemoryRequirements bufferMemReqs[2];
		vkGetBufferMemoryRequirements(device, buffer, &bufferMemReqs[0]);
		vkGetBufferMemoryRequirements(device, unifsBuffer, &bufferMemReqs[1]);
		const size_t unifsBufferOffset = aligned(bufferMemReqs[0].size, bufferMemReqs[1].alignment);
		const size_t memSize = unifsBufferOffset + bufferMemReqs[1].size;

		//VkMemoryPropertyFlagBits memPropFlags;
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = memSize;
		allocInfo.memoryTypeIndex = localMemTypeInd;
		vkAllocateMemory(device, &allocInfo, nullptr, &bufferMem);
		assert(bufferMem && "Error allocating memory");
		vkBindBufferMemory(device, buffer, bufferMem, 0);
		vkBindBufferMemory(device, unifsBuffer, bufferMem, unifsBufferOffset);
	}

	VkDeviceMemory stagingBufferMem = nullptr;
	{
		VkMemoryRequirements bufferMemReqs;
		vkGetBufferMemoryRequirements(device, stagingBuffer, &bufferMemReqs);
		VkMemoryAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
		allocInfo.allocationSize = bufferMemReqs.size;
		allocInfo.memoryTypeIndex = hostVisibleMemTypeInd;
		vkAllocateMemory(device, &allocInfo, nullptr, &stagingBufferMem);
		vkBindBufferMemory(device, stagingBuffer, stagingBufferMem, 0);
	}

	VkShaderModule shaderModule;
	{
		VkShaderModuleCreateInfo shaderModuleCreateInfo = {};
		FILE *file = fopen("../calc.spv", "rb");
		const int MAX_CODE_SIZE = 1 << 20;
		u32 *code = new u32[MAX_CODE_SIZE];
		const int codeSize = fread(code, sizeof(u32), MAX_CODE_SIZE - 1, file);
		shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		shaderModuleCreateInfo.pCode = code;
		shaderModuleCreateInfo.codeSize = 4 * codeSize;
		vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);
		delete[] code;
		fclose(file);
	}

	VkPipelineShaderStageCreateInfo stageCreateInfo = {};
	stageCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
	stageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
	stageCreateInfo.module = shaderModule;
	stageCreateInfo.pName = "main";
	//info.pSpecializationInfo = nullptr;

	VkDescriptorSetLayout* descriptorSetLayout = new VkDescriptorSetLayout[2];
	// TODO: delete[]
	{
		VkDescriptorSetLayoutBinding descriptorSetLayoutBinding;
		descriptorSetLayoutBinding.binding = 0;
		descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		descriptorSetLayoutBinding.descriptorCount = 1;
		descriptorSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo = {};
		descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		descriptorSetLayoutCreateInfo.bindingCount = 1;
		descriptorSetLayoutCreateInfo.pBindings = &descriptorSetLayoutBinding;
		vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout[0]);

		descriptorSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout[1]);
	}

	VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo = {};
	pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
	pipelineLayoutCreateInfo.setLayoutCount = 2;
	pipelineLayoutCreateInfo.pSetLayouts = descriptorSetLayout;

	VkPipelineLayout pipelineLayout;
	vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

	VkComputePipelineCreateInfo pipelineCreateInfo = {};
	pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
	pipelineCreateInfo.stage = stageCreateInfo;
	pipelineCreateInfo.layout = pipelineLayout;

	VkPipeline pipeline;
	vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);

	VkDescriptorPool descriptorPool;
	{
		VkDescriptorPoolSize poolSizes[2] = {};
		poolSizes[0].type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		poolSizes[0].descriptorCount = 1;
		poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		poolSizes[1].descriptorCount = 1;
		VkDescriptorPoolCreateInfo descriptorPoolCreateInfo = {};
		descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		descriptorPoolCreateInfo.maxSets = 2;
		descriptorPoolCreateInfo.poolSizeCount = 2;
		descriptorPoolCreateInfo.pPoolSizes = poolSizes;
		vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);
	}

	VkDescriptorSet* descriptorSets = new VkDescriptorSet[2];
	{
		VkDescriptorSetAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		allocInfo.descriptorPool = descriptorPool;
		allocInfo.descriptorSetCount = 2;
		allocInfo.pSetLayouts = descriptorSetLayout;
		vkAllocateDescriptorSets(device, &allocInfo, descriptorSets);

		VkDescriptorBufferInfo bufferInfos[2];
		bufferInfos[0].buffer = unifsBuffer;
		bufferInfos[0].offset = 0;
		bufferInfos[0].range = VK_WHOLE_SIZE;
		bufferInfos[1].buffer = buffer;
		bufferInfos[1].offset = 0;
		bufferInfos[1].range = VK_WHOLE_SIZE;

		VkWriteDescriptorSet writes[2] = {};
		writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[0].dstSet = descriptorSets[0];
		writes[0].dstBinding = 0;
		writes[0].dstArrayElement = 0;
		writes[0].descriptorCount = 1;
		writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
		writes[0].pBufferInfo = &bufferInfos[0];
		writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		writes[1].dstSet = descriptorSets[1];
		writes[1].dstBinding = 0;
		writes[1].dstArrayElement = 0;
		writes[1].descriptorCount = 1;
		writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writes[1].pBufferInfo = &bufferInfos[1];
		vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);
	}

	VkCommandPool commandPool;
	{
		VkCommandPoolCreateInfo info = {};
		info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
		info.queueFamilyIndex = bestQueueFamilyInd;
		vkCreateCommandPool(device, &info, nullptr, &commandPool);
	}
	
	VkCommandBuffer commandBuffer;
	{
		VkCommandBufferAllocateInfo allocInfo = {};
		allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
		allocInfo.commandPool = commandPool;
		allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
		allocInfo.commandBufferCount = 1;
		vkAllocateCommandBuffers(device, &allocInfo, &commandBuffer);
	}

	VkBufferMemoryBarrier* memBarriers = new VkBufferMemoryBarrier[3];
	memBarriers[0] = {};
	memBarriers[0].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	memBarriers[0].srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
	memBarriers[0].dstAccessMask = VK_ACCESS_UNIFORM_READ_BIT;
	memBarriers[0].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[0].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[0].buffer = unifsBuffer;
	memBarriers[0].offset = 0;
	memBarriers[0].size = VK_WHOLE_SIZE;

	memBarriers[1] = {};
	memBarriers[1].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	memBarriers[1].srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
	memBarriers[1].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	memBarriers[1].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[1].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[1].buffer = buffer;
	memBarriers[1].offset = 0;
	memBarriers[1].size = VK_WHOLE_SIZE;

	memBarriers[2] = {};
	memBarriers[2].sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
	memBarriers[2].srcAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	memBarriers[2].dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
	memBarriers[2].srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[2].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
	memBarriers[2].buffer = stagingBuffer;
	memBarriers[2].offset = 0;
	memBarriers[2].size = VK_WHOLE_SIZE;

	VkCommandBufferBeginInfo beginInfo = {};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	beginInfo.flags = 0;
	beginInfo.pInheritanceInfo = nullptr;
	vkBeginCommandBuffer(commandBuffer, &beginInfo);

	vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
	vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
		0, 2, descriptorSets, 0, nullptr);

	const double startT = getTime();
	i64 percent = 0;
	for (i64 i = 0; i < numIterations; i++)
	{
		i64 start = i * perIteration;
		i64 newPercent = (10000 * start) / N;
		if (newPercent/1 > percent/1)
		{
			printf("%" PRId64 ".%" PRId64 "%%\n", newPercent / 100, newPercent % 100);
			percent = newPercent;
			calcElapsedAndPrintETA(startT, i, numIterations);
		}

		if (i != 0)
			vkBeginCommandBuffer(commandBuffer, &beginInfo);

		uniforms.start = i * numThreads;
		vkCmdUpdateBuffer(commandBuffer, unifsBuffer, 0, sizeof(uniforms), &uniforms);
		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0,
			0, nullptr,
			1, &memBarriers[0],
			0, nullptr);

		vkCmdDispatch(commandBuffer, numThreads, 1, 1);

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			1, &memBarriers[1],
			0, nullptr);

		VkBufferCopy copyInfo = {};
		copyInfo.srcOffset = 0;
		copyInfo.dstOffset = 0;
		copyInfo.size = sizeof(i64) * numThreads;
		vkCmdCopyBuffer(commandBuffer,
			buffer, stagingBuffer, 1, &copyInfo);

		vkCmdPipelineBarrier(commandBuffer,
			VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
			0, nullptr,
			1, &memBarriers[2],
			0, nullptr);

		vkEndCommandBuffer(commandBuffer);
		
		{
			VkSubmitInfo info = {};
			info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
			info.commandBufferCount = 1;
			info.pCommandBuffers = &commandBuffer;
			vkQueueSubmit(queue, 1, &info, VK_NULL_HANDLE);
		}
		vkQueueWaitIdle(queue);

		i64* result;
		vkMapMemory(device, stagingBufferMem, 0, sizeof(i64) * numThreads, 0, (void**)&result);

		for (int i = 0; i < numThreads; i++)
		{
			if (result[i])
				return result[i];
		}
		vkUnmapMemory(device, stagingBufferMem);
	}
 
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