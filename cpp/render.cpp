#include <iostream>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <cstdlib>
#include <set>
#include <string>

#include "constant.h"
#include "core.h"
#include "Shader.h"
#include "LineStrip.h"
#include <complex>
#include <string>
#include <thread>

#include "NumCpp.hpp"
#include "FourierSeries.h"
#include "RenderParam.h"
#include "RenderInstance.h"
#include "bsem.h"

#include "profile.h"

#if COMPILE_CUDA
#include <cuda_runtime.h>
#include "CudaFourierSeries.cuh"
#else
#include "NpFourierSeries.h"
#endif



//https://stackoverflow.com/a/26221725
template<typename ... Args>
std::string string_format(const std::string& format, Args ... args)
{
	int size_s = std::snprintf(nullptr, 0, format.c_str(), args ...) + 1; // Extra space for '\0'
	if (size_s <= 0) { throw std::runtime_error("Error during formatting."); }
	auto size = static_cast<size_t>(size_s);
	std::unique_ptr<char[]> buf(new char[size]);
	std::snprintf(buf.get(), size, format.c_str(), args ...);
	return std::string(buf.get(), buf.get() + size - 1); // We don't want the '\0' inside
}

std::string formatTime(int seconds)
{
	if (seconds > 3599)
		return string_format("%d:%02d:%02d", seconds / 3600, (seconds % 3600) / 60, seconds % 60);
	return string_format("%02d:%02d", seconds / 60, seconds % 60);
}
std::string formatTime(double seconds)
{
	if (seconds < 0.1)
	{
		seconds *= 1000;
		if (seconds > 0.1)
			return string_format("%.3fms", seconds);
		seconds *= 1000;
		if (seconds > 0.1)
			return string_format("%.3fus", seconds);
		seconds *= 1000;
		if (seconds > 0.1)
			return string_format("%.3fns", seconds);
	}
	else
	{
		int sec = (int)seconds;
		if (sec > 3599)
			return string_format("%d:%02d:%02d.%03d", sec / 3600, (sec % 3600) / 60, sec % 60, (int)((seconds - sec) * 1000));
		return string_format("%02d:%02d.%03d", sec / 60, sec % 60, (int)((seconds - sec) * 1000));
	}
	return "";
}

volatile bool alive = true;
void keyboard_interrupt(int signum) {
	alive = false;
}

char buf[256];
int printProgressBar(float part, int barLength = 40, int minLength = 0, std::string prefix = "", std::string suffix = "")
{
	std::string bar = std::string((int)(barLength * part), '*');
	bar += std::string(barLength - bar.length(), ' ');
	if (part > 1)
		part = 1;
	int ret = sprintf(buf, "%s |%s| %.1f%% %s", prefix.c_str(), bar.c_str(), part * 100, suffix.c_str());
	printf("%-*s\r", minLength, buf);
	return ret;
}

/*
std::string severityConversion(GLenum severity) {
	switch (severity) {
	case GL_DEBUG_SEVERITY_NOTIFICATION:
		return "notification";
	case GL_DEBUG_SEVERITY_LOW:
		return "low";
	case GL_DEBUG_SEVERITY_MEDIUM:
		return "medium";
	case GL_DEBUG_SEVERITY_HIGH:
		return "high";
	default:
		return "";
	}
}
void GLAPIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = %s (0x%x), message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		type, severityConversion(severity).c_str(), severity, message);
}
void GLAPIENTRY warnCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	if (severity > GL_DEBUG_SEVERITY_NOTIFICATION)
		fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = %s (0x%x), message = %s\n",
			(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
			type, severityConversion(severity).c_str(), severity, message);
}
*/
void APIENTRY glDebugOutput(GLenum source,
	GLenum type,
	unsigned int id,
	GLenum severity,
	GLsizei length,
	const char* message,
	const void* userParam)
{
	// ignore non-significant error/warning codes
	if (id == 131169 || id == 131185 || id == 131218 || id == 131204) return;

	std::cout << "---------------" << std::endl;
	std::cout << "Debug message (" << id << "): " << message << std::endl;

	switch (source)
	{
	case GL_DEBUG_SOURCE_API:             std::cout << "Source: API"; break;
	case GL_DEBUG_SOURCE_WINDOW_SYSTEM:   std::cout << "Source: Window System"; break;
	case GL_DEBUG_SOURCE_SHADER_COMPILER: std::cout << "Source: Shader Compiler"; break;
	case GL_DEBUG_SOURCE_THIRD_PARTY:     std::cout << "Source: Third Party"; break;
	case GL_DEBUG_SOURCE_APPLICATION:     std::cout << "Source: Application"; break;
	case GL_DEBUG_SOURCE_OTHER:           std::cout << "Source: Other"; break;
	} std::cout << std::endl;

	switch (type)
	{
	case GL_DEBUG_TYPE_ERROR:               std::cout << "Type: Error"; break;
	case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR: std::cout << "Type: Deprecated Behaviour"; break;
	case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR:  std::cout << "Type: Undefined Behaviour"; break;
	case GL_DEBUG_TYPE_PORTABILITY:         std::cout << "Type: Portability"; break;
	case GL_DEBUG_TYPE_PERFORMANCE:         std::cout << "Type: Performance"; break;
	case GL_DEBUG_TYPE_MARKER:              std::cout << "Type: Marker"; break;
	case GL_DEBUG_TYPE_PUSH_GROUP:          std::cout << "Type: Push Group"; break;
	case GL_DEBUG_TYPE_POP_GROUP:           std::cout << "Type: Pop Group"; break;
	case GL_DEBUG_TYPE_OTHER:               std::cout << "Type: Other"; break;
	} std::cout << std::endl;

	switch (severity)
	{
	case GL_DEBUG_SEVERITY_HIGH:         std::cout << "Severity: high"; break;
	case GL_DEBUG_SEVERITY_MEDIUM:       std::cout << "Severity: medium"; break;
	case GL_DEBUG_SEVERITY_LOW:          std::cout << "Severity: low"; break;
	case GL_DEBUG_SEVERITY_NOTIFICATION: std::cout << "Severity: notification"; break;
	} std::cout << std::endl;
	std::cout << std::endl;
}

GLFWwindow* createSharedWindow(GLFWwindow* mainWindow) {
	glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
	GLFWwindow* window = glfwCreateWindow(1, 1, "Shared", NULL, mainWindow);
	if (!window) {
		std::cerr << "Failed to open GLFW window." << std::endl;
		glfwTerminate();
		exit(-1);
	}

	glfwMakeContextCurrent(window);
	glewInit();
	return window;
}

#define TIMEOUT 30000000000ul
extern "C" {
	DLL_API int __cdecl render(float* data, size_t size, int width, int height, float dt, float duration, float start,
		float pathLength, RenderParam* renders, size_t renderCount, int spf, int gpu, bool show, int flags)
	{
		std::cout << "Initializing Scene" << std::endl;
		signal(SIGINT, keyboard_interrupt);

		if (!glfwInit()) {
			std::cerr << "Failed to initialize GLFW" << std::endl;
			return -1;
		}

		glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
		GLFWwindow* window = glfwCreateWindow(1, 1, "Fourier", NULL, NULL);
		if (!window) {
			std::cerr << "Failed to open GLFW window." << std::endl;
			glfwTerminate();
			return -1;
		}

		glfwMakeContextCurrent(window);
		glewInit();
		glfwSwapInterval(1);

		glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_FALSE);
		if (flags & DEBUG_FLAG)
		{
			glEnable(GL_DEBUG_OUTPUT);
			glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
			glDebugMessageCallback(glDebugOutput, nullptr);
			glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DEBUG_TYPE_ERROR, GL_DONT_CARE, 0, nullptr, GL_TRUE);
		}
		else if (flags & WARN_FLAG)
		{
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageControl(GL_DEBUG_SOURCE_API, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr, GL_TRUE);
		}

		bool hasFade = false;
		for (int i = 0; i < renderCount; i++)
			for (int j = 0; j < 8; j++)
				if (renders[i].views[i].valid && renders[i].views[j].path_fade)
					hasFade = true;

		GLuint solidShader = LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/solid.frag", flags & DEBUG_FLAG);
		GLuint fadeShader = hasFade ? LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/fade.frag", flags & DEBUG_FLAG) : 0;
		if (!solidShader || (hasFade && !fadeShader)) {
			std::cerr << "Failed to initialize shader program" << std::endl;
			glfwDestroyWindow(window);
			glfwTerminate();
			return -1;
		}

		size_t vectorSize = size / 2;
		size_t pathSize = (size_t)(pathLength / dt);

		if (hasFade)
		{
			glUseProgram(fadeShader);
			glUniform1f(glGetUniformLocation(fadeShader, "pathLength"), pathLength);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		LineStrip* vector = new LineStrip(glm::vec2(0,0), vectorSize);
		Lines* path = new Lines(glm::vec2(0,0), pathSize, hasFade);
		
		nc::NdArray<std::complex<float>> mags((std::complex<float>*)data, vectorSize);
		nc::NdArray<int> freqs = nc::append(nc::arange(0, (int)(vectorSize / 2)), nc::arange(-(int)((vectorSize + 1) / 2), 0));
		nc::NdArray<nc::uint32> inds = nc::argsort(-nc::abs(mags));
		mags = mags[inds];
		freqs = freqs[inds];
		
#if COMPILE_CUDA
		FourierSeries* fourier = new CudaFourierSeries(vector, path, mags.dataRelease(), freqs.dataRelease(), vectorSize, dt, spf, gpu, flags & GPU_FLAG);
#else
		FourierSeries* fourier = new NpForuierSeries(vector, path, mags.dataRelease(), freqs.dataRelease(), vectorSize, dt, spf);
#endif
		if (!fourier->valid())
		{
			alive = false;
		}

		glEnable(GL_STENCIL_TEST);
		glDisable(GL_DEPTH_TEST);
		std::set<std::string> names;
		for (int i = 0; i < renderCount; i++)
		{
			std::string n = renders[i].output_name;
			std::string nn = n;
			int k = 0;
			while (names.find(nn) != names.end())
			{
				int ind = n.find_last_of('.');
				nn = n.substr(0, ind) + std::to_string(++k) + n.substr(ind);
			}
			names.insert(nn);
			char* nnp = (char*)malloc(sizeof(char) * (nn.size() + 1));
			strcpy(nnp, nn.c_str());
			renders[i].output_name = nnp;
		}

		std::vector<RenderInstance*> renderInstances;
		for (int i = 0; i < renderCount; i++)
		{
			if (flags & RENDER_FLAG)
				std::cout << "Instance" << std::to_string(i+1) << ": " << renders[i] << std::endl;
			renderInstances.push_back(new RenderInstance(renders[i], solidShader, fadeShader, vector, path, width, height));
		}

		glClearColor(0, 0, 0, 1);
		float t = start;
		float end = start + duration;
		if (t > 0)
		{
			fourier->init(t);
			fourier->updateBuffers();
		}
		fourier->resetTrail();

		std::string ETR = "XX:XX remaining";
		int ind = 0;
		double sTime = glfwGetTime(), tTime = -1;
		double pTime = glfwGetTime();
		double rt64[64] = { 0 };
		double st64[64] = { 0 };
		int len = 0;
		GLsync copy, step, draw;
		GLsync* draws = (GLsync*)malloc(sizeof(GLsync) * renderCount);
		glm::vec2* vecHead = nullptr;
		for (int i = 0; i < renderCount; i++)
			for (int j = 0; j < 8; j++)
				if (renders[i].views[i].valid && renders[i].views[j].follow_path)
				{
					vecHead = new glm::vec2(0, 0);
					break;
				}
		if (flags & PROFILE_FLAG)
		{
			std::mutex contextLock, etrLock;
			bsem computeSem(false), encodeSem(true), drawSem(false), copySem(false);
			std::thread renderThread, computeThread, encodeThread, sampleThread, printThread;

			MAKE_RANGE(render_rid);
			MAKE_RANGE(step_rid);
			MAKE_RANGE(encode_rid);
			MAKE_RANGE(copy_rid);
			glfwMakeContextCurrent(nullptr);

#if COMPILE_CUDA
			cudaEvent_t stepEvent;
			cudaEventCreate(&stepEvent);
#endif
			renderThread = std::thread([&]() {
				while (t < end && alive) {
					END_RANGE(copy_rid);
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						START_RANGE(render_rid, "render", GREEN);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->draw(t, vecHead);
						draw = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
						glfwMakeContextCurrent(nullptr);
					}
					encodeSem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->postDraw();
						copy = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
						glfwMakeContextCurrent(nullptr);
					}
					drawSem.release();
					computeSem.acquire();
				}
			});
			computeThread = std::thread([&]() {
				while (t < end && alive) {
					START_RANGE(step_rid, "step", AQUA);
					t += fourier->increment(spf, t);
#if COMPILE_CUDA
					cudaEventRecord(stepEvent, 0);
					cudaStreamAddCallback(0, [](cudaStream_t stream, cudaError_t status, void* userData) {
						END_RANGE(*((nvtxRangeId_t*)userData));
					}, &step_rid, 0);
#else
					END_RANGE(step_rid);
#endif
					drawSem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						glClientWaitSync(draw, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
						END_RANGE(render_rid);
						START_RANGE(copy_rid, "copy", YELLOW);
						fourier->updateBuffers(vecHead);
						fourier->readyBuffers();
						glfwMakeContextCurrent(nullptr);
					}
					computeSem.release();
					copySem.release();
				}
			});
			encodeThread = std::thread([&]() {
				while (t < end && alive) {
					copySem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						glClientWaitSync(copy, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
						END_RANGE(copy_rid);
						START_RANGE(encode_rid, "encode", RED);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->encode();
						END_RANGE(encode_rid);
						glfwMakeContextCurrent(nullptr);
					}
					encodeSem.release();
				}
			});
			sampleThread = std::thread([&]() {
				double time = glfwGetTime();
				float lastT = t;
				while (t < end && alive) {
					std::this_thread::sleep_for(std::chrono::seconds(1));
					rt64[ind] = (glfwGetTime() - time);
					st64[ind] = (t - lastT);
					ind = (ind + 1) & 0b111111;
					pTime = glfwGetTime();
					double rtsum = 0;
					double stsum = 0;
					for (double d : rt64)
						rtsum += d;
					for (double d : st64)
						stsum += d;
					{
						std::lock_guard<std::mutex> guard(etrLock);
						ETR = formatTime((int)(rtsum / stsum * (end - t))) + " remaining";
					}
					time = glfwGetTime();
					lastT = t;
				}				
			});
			printThread = std::thread([&]() {
				{
					std::lock_guard<std::mutex> guard(etrLock);
					len = printProgressBar(0, 40, len, "Rendering:", ETR);
				}
				while (t < end && alive) {
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					{
						std::lock_guard<std::mutex> guard(etrLock);
						len = printProgressBar((t - start) / duration, 40, len, "Rendering:", ETR);
					}
				}
			});

			renderThread.join();
			computeThread.join();
			encodeThread.join();
			printThread.join();
			sampleThread.join();
			tTime = glfwGetTime() - sTime;
			END_RANGE(render_rid);
			END_RANGE(copy_rid);
			glfwMakeContextCurrent(window);
		}
		else
		{
			std::mutex contextLock, etrLock;
			bsem computeSem(false), encodeSem(true), drawSem(false), copySem(false);
			std::thread renderThread, computeThread, encodeThread, sampleThread, printThread;

			glfwMakeContextCurrent(nullptr);

			renderThread = std::thread([&]() {
				while (t < end && alive) {
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->draw(t, vecHead);
						draw = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
						glfwMakeContextCurrent(nullptr);
					}
					encodeSem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->postDraw();
						copy = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
						glfwMakeContextCurrent(nullptr);
					}
					drawSem.release();
					computeSem.acquire();
				}
				});
			computeThread = std::thread([&]() {
				while (t < end && alive) {
					t += fourier->increment(spf, t);
					drawSem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						glClientWaitSync(draw, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
						fourier->updateBuffers(vecHead);
						fourier->readyBuffers();
						glfwMakeContextCurrent(nullptr);
					}
					computeSem.release();
					copySem.release();
				}
				});
			encodeThread = std::thread([&]() {
				while (t < end && alive) {
					copySem.acquire();
					{
						std::lock_guard<std::mutex> guard(contextLock);
						glfwMakeContextCurrent(window);
						glClientWaitSync(copy, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
						for (int i = 0; i < renderCount; i++)
							renderInstances[i]->encode();
						glfwMakeContextCurrent(nullptr);
					}
					encodeSem.release();
				}
				});
			sampleThread = std::thread([&]() {
				double time = glfwGetTime();
				float lastT = t;
				while (t < end && alive) {
					std::this_thread::sleep_for(std::chrono::seconds(1));
					rt64[ind] = (glfwGetTime() - time);
					st64[ind] = (t - lastT);
					ind = (ind + 1) & 0b111111;
					pTime = glfwGetTime();
					double rtsum = 0;
					double stsum = 0;
					for (double d : rt64)
						rtsum += d;
					for (double d : st64)
						stsum += d;
					{
						std::lock_guard<std::mutex> guard(etrLock);
						ETR = formatTime((int)(rtsum / stsum * (end - t))) + " remaining";
					}
					time = glfwGetTime();
					lastT = t;
				}
				});
			printThread = std::thread([&]() {
				{
					std::lock_guard<std::mutex> guard(etrLock);
					len = printProgressBar(0, 40, len, "Rendering:", ETR);
				}
				while (t < end && alive) {
					std::this_thread::sleep_for(std::chrono::milliseconds(10));
					{
						std::lock_guard<std::mutex> guard(etrLock);
						len = printProgressBar((t - start) / duration, 40, len, "Rendering:", ETR);
					}
				}
				});

			renderThread.join();
			computeThread.join();
			encodeThread.join();
			printThread.join();
			sampleThread.join();
			tTime = glfwGetTime() - sTime;
			glfwMakeContextCurrent(window);
		}
		
		if (alive)
			std::cout << std::endl << "Total Time: " << formatTime(tTime) << std::endl;
		else
			std::cout << std::endl << "Program Terminated" << std::endl;
		if(vecHead)
			delete vecHead;

		delete vector;
		delete path;
		if(fourier != nullptr)
			delete fourier;
		for (int i = 0; i < renderCount; i++)
		{
			free(renders[i].output_name);
			delete renderInstances[i];
		}

		glDeleteProgram(solidShader);
		if (hasFade)
			glDeleteProgram(fadeShader);

		glfwDestroyWindow(window);
		glfwTerminate();
		return 0;
	}
}
