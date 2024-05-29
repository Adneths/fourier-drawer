#pragma once
#define GLEW_STATIC
#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>

namespace glutil {
	template <typename T>
	void glUniformMatrix3v(GLint location, GLsizei count, GLboolean transpose, const T* value) {
		if constexpr (std::is_same_v<T, float>)
			glUniformMatrix3fv(location, count, transpose, (GLfloat*)value);
		else if constexpr (std::is_same_v<T, double>)
			glUniformMatrix3dv(location, count, transpose, (GLdouble*)value);
	}
	template <typename T>
	void glUniform2v(GLint location, GLsizei count, const T* value) {
		if constexpr (std::is_same_v<T, float>)
			glUniform2fv(location, count, (GLfloat*)value);
		else if constexpr (std::is_same_v<T, double>)
			glUniform2dv(location, count, (GLdouble*)value);
	}
	template <typename T>
	void glUniform1(GLint location, const T value) {
		if constexpr (std::is_same_v<T, float>)
			glUniform1f(location, (GLfloat)value);
		else if constexpr (std::is_same_v<T, double>)
			glUniform1d(location, (GLdouble)value);
	}
	template <typename T>
	void glVertexAttribPtr(GLuint index, GLint size, GLboolean normalized, GLsizei stride, const void* pointer) {
		if constexpr (std::is_same_v<T, float>)
			glVertexAttribPointer(index, size, GL_FLOAT, normalized, stride, pointer);
		else if constexpr (std::is_same_v<T, double>)
			glVertexAttribLPointer(index, size, GL_DOUBLE, stride, pointer);
	}

	template <typename T>
	void fill_buffer(GLuint shader, GLuint buffer, T* val, unsigned int size, unsigned int offset, unsigned int length) {
		if (size * sizeof(T) / sizeof(unsigned int) > 12) {
			exit(-1);
		}
		glUseProgram(shader);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, buffer);
		unsigned int buf[12];
		unsigned int* uival = (unsigned int*)val;
		for (unsigned int i = 0; i < size * sizeof(T) / sizeof(unsigned int); i++)
			buf[i] = uival[i];
		glUniform4uiv(0, 4, buf);
		glUniform4uiv(1, 4, buf+4);
		glUniform4uiv(2, 4, buf+8);
		glUniform1ui(3, size * sizeof(T) / sizeof(unsigned int));
		glUniform1ui(4, offset);
		glUniform1ui(5, length * sizeof(T) / sizeof(unsigned int));

		glDispatchCompute(((length+1023)/1024)%32768, (((length+1023)/1024+32767)/32768)%32768, (((length+1023)/1024+32767)/32768+32767)/32768);
		glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
		glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0);
		glUseProgram(0);
	}
};

namespace math {
	template <typename T> using vec2 = glm::vec<2, T, glm::defaultp>;
	template <typename T> using vec3 = glm::vec<3, T, glm::defaultp>;
	template <typename T> using vec4 = glm::vec<4, T, glm::defaultp>;
	template <typename T> using mat2 = glm::mat<2, 2, T, glm::defaultp>;
	template <typename T> using mat3 = glm::mat<3, 3, T, glm::defaultp>;
	template <typename T> using mat4 = glm::mat<4, 4, T, glm::defaultp>;

	using fvec2 = glm::vec2;
	using fvec3 = glm::vec3;
	using fvec4 = glm::vec4;
	using fmat2 = glm::mat2;
	using fmat3 = glm::mat3;
	using fmat4 = glm::mat4;

	using dvec2 = glm::dvec2;
	using dvec3 = glm::dvec3;
	using dvec4 = glm::dvec4;
	using dmat2 = glm::dmat2;
	using dmat3 = glm::dmat3;
	using dmat4 = glm::dmat4;

	template <typename T>
	constexpr GLenum datatype() {
		if constexpr (std::is_same_v<T, float>)
			return GL_FLOAT;
		else if constexpr (std::is_same_v<T, double>)
			return GL_DOUBLE;
		else
			return GL_ZERO;
	}
}