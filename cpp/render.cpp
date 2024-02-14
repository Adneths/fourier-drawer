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

#include "NumCpp.hpp"
#include "FourierSeries.h"
#include "RenderParam.h"
#include "RenderInstance.h"

#if PROFILE
#include "profile.h"
#endif

#if COMPILE_CUDA
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

bool alive = true;
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

void GLAPIENTRY errorCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
		(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
		type, severity, message);
}


#define TIMEOUT 30000000000ul
extern "C" {
	DLL_API int __cdecl render(float* data, size_t size, int width, int height, float dt, float duration, float start,
		float trailLength, RenderParam* renders, size_t renderCount, int fpf, int gpu, bool show, int flags)
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

		if (flags & DEBUG_FLAG)
		{
			glEnable(GL_DEBUG_OUTPUT);
			glDebugMessageCallback(errorCallback, 0);
		}

		bool hasFade = false;
		for (int i = 0; i < renderCount; i++)
			if (renders[i].trailFade)
				hasFade = true;

		GLuint vectorShader = LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/solid.frag", flags & DEBUG_FLAG);
		GLuint fadeShader = hasFade ? LoadShaders("./libs/shaders/2d.vert", "./libs/shaders/fade.frag", flags & DEBUG_FLAG) : 0;
		if (!vectorShader) {
			std::cerr << "Failed to initialize shader program" << std::endl;
			glfwDestroyWindow(window);
			glfwTerminate();
			return -1;
		}

		size_t vectorSize = size / 2;
		size_t trailSize = (size_t)(trailLength / dt);

		if (hasFade)
		{
			glUseProgram(fadeShader);
			glUniform1f(glGetUniformLocation(fadeShader, "trailLength"), trailLength);
			glEnable(GL_BLEND);
			glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		}

		LineStrip* vector = new LineStrip(glm::vec2(0,0), vectorSize);
		Lines* trail = new Lines(glm::vec2(0,0), trailSize, hasFade);
		

		nc::NdArray<std::complex<float>> mags((std::complex<float>*)data, vectorSize);
		nc::NdArray<int> freqs = nc::append(nc::arange(0, (int)(vectorSize / 2)), nc::arange(-(int)((vectorSize + 1) / 2), 0));
		nc::NdArray<nc::uint32> inds = nc::argsort(-nc::abs(mags));
		mags = mags[inds];
		freqs = freqs[inds];
		
#if COMPILE_CUDA
		FourierSeries* fourier = new CudaFourierSeries(vector, trail, mags.dataRelease(), freqs.dataRelease(), vectorSize, dt, fpf, gpu, flags & GPU_FLAG);
#else
		FourierSeries* fourier = new NpForuierSeries(vector, trail, mags.dataRelease(), freqs.dataRelease(), vectorSize, dt, fpf);
#endif
		if (!fourier->valid())
		{
			alive = false;
		}

		std::set<std::string> names;
		for (int i = 0; i < renderCount; i++)
		{
			std::string n = renders[i].output;
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
			renders[i].output = nnp;
		}

		std::vector<RenderInstance*> renderInstances;
		for (int i = 0; i < renderCount; i++)
		{
			if (flags & RENDER_FLAG)
				std::cout << "Instance" << std::to_string(i+1) << ": " << renders[i] << std::endl;
			renderInstances.push_back(new RenderInstance(renders[i], vectorShader, renders[i].trailFade ? fadeShader : vectorShader, vector, trail, width, height));
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
		double d64[64] = {0};
		int len = 0;
		GLsync copy, step;
		GLsync* draws = (GLsync*)malloc(sizeof(GLsync) * renderCount);
		glm::vec2* vecHead = nullptr;
		for (int i = 0; i < renderCount; i++)
			if (renders[i].followTrail)
			{
				vecHead = new glm::vec2(0, 0);
				break;
			}
		if (flags & PROFILE_FLAG)
		{
			size_t fCount = 0;
			double renderD = 0, stepD = 0, encodeD = 0;
			while (t < end && alive) {
				fCount++;
				double time = glfwGetTime();

				PUSH_RANGE("render", GREEN);
				MEASURE(renderD) {
					for (int i = 0; i < renderCount; i++)
						draws[i] = renderInstances[i]->draw(t, renderInstances[i]->params.followTrail ? vecHead : nullptr);
					copy = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
					for (int i = 0; i < renderCount; i++)
						glClientWaitSync(draws[i], GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
					glClientWaitSync(copy, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
				}
				POP_RANGE();

				PUSH_RANGE("step", AQUA);
				MEASURE(stepD) {
					t += fourier->increment(fpf, t);
					fourier->updateBuffers();
					fourier->readyBuffers(vecHead);
					step = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
					glClientWaitSync(step, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
				}
				POP_RANGE();

				PUSH_RANGE("encode", RED);
				MEASURE(encodeD) {
					for (int i = 0; i < renderCount; i++)
						renderInstances[i]->encode();
				}
				POP_RANGE();


				d64[(ind = (ind + 1) & 0b111111)] = glfwGetTime() - time;
				if (glfwGetTime() - pTime > 2)
				{
					pTime = glfwGetTime();
					double sum = 0;
					for (double d : d64)
						sum += d;
					sum /= 64 * fpf;
					ETR = formatTime((int)(sum * (end - t) / dt)) + " remaining";
				}

				len = printProgressBar((t - start) / duration, 40, len, "Rendering:", ETR);
			}
			tTime = glfwGetTime() - sTime;
			if (alive)
			{
				double otherD = tTime - renderD - stepD - encodeD;
				printf("\n");
				printf("Type      Percent           Total         Frame\n-----------------------------------------------\n");
				printf("Total      %6.2f  %14s  %12s\n", 100.0f, formatTime(tTime).c_str(), formatTime(tTime / fCount).c_str());
				printf("Render     %6.2f  %14s  %12s\n", 100 * renderD / tTime, formatTime(renderD).c_str(), formatTime(renderD / fCount).c_str());
				printf("Step       %6.2f  %14s  %12s\n", 100 * stepD / tTime, formatTime(stepD).c_str(), formatTime(stepD / fCount).c_str());
				printf("Encode     %6.2f  %14s  %12s\n", 100 * encodeD / tTime, formatTime(encodeD).c_str(), formatTime(encodeD / fCount).c_str());
				printf("Other      %6.2f  %14s  %12s", 100 * otherD / tTime, formatTime(otherD).c_str(), formatTime(otherD / fCount).c_str());
			}
		}
		else
		{
			while (t < end && alive) {
				//glfwPollEvents();
				//glfwSwapBuffers(window);

				fourier->readyBuffers(vecHead);
				double time = glfwGetTime();
				for (int i = 0; i < renderCount; i++)
					draws[i] = renderInstances[i]->draw(t, renderInstances[i]->params.followTrail ? vecHead : nullptr);
				copy = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);


				t += fourier->increment(fpf, t);
				step = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

				for (int i = 0; i < renderCount; i++)
					glClientWaitSync(draws[i], GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
				glClientWaitSync(step, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
				fourier->updateBuffers();


				glClientWaitSync(copy, GL_SYNC_FLUSH_COMMANDS_BIT, TIMEOUT);
				for (int i = 0; i < renderCount; i++)
					renderInstances[i]->encode();

				d64[(ind = (ind + 1) & 0b111111)] = glfwGetTime() - time;
				if (glfwGetTime() - pTime > 2)
				{
					pTime = glfwGetTime();
					double sum = 0;
					for (double d : d64)
						sum += d;
					sum /= 64 * fpf;
					ETR = formatTime((int)(sum * (end - t) / dt)) + " remaining";
				}

				len = printProgressBar((t - start) / duration, 40, len, "Rendering:", ETR);
			}
			tTime = glfwGetTime() - sTime;
		}
		if (alive)
			std::cout << std::endl << "Total Time: " << formatTime(tTime) << std::endl;
		else
			std::cout << std::endl << "Program Terminated" << std::endl;
		if(vecHead)
			delete vecHead;

		delete vector;
		delete trail;
		if(fourier != nullptr)
			delete fourier;
		for (int i = 0; i < renderCount; i++)
		{
			free(renders[i].output);
			delete renderInstances[i];
		}

		glDeleteProgram(vectorShader);
		if (hasFade)
			glDeleteProgram(fadeShader);

		glfwDestroyWindow(window);
		glfwTerminate();
		return 0;
	}
}
