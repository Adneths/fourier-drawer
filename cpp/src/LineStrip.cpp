#include "LineStrip.h"
template class LineStrip<float>;
template class LineStrip<double>;

template <typename T>
LineStrip<T>::LineStrip(vec2<T> vertex, uint32_t count)
{
	this->count = count;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 2 * (count+1) * sizeof(T), nullptr, GL_STREAM_DRAW);
	if constexpr (std::is_same_v<T, float>)
		glClearBufferData(GL_ARRAY_BUFFER, GL_RG32F, GL_RGBA, GL_FLOAT, &vertex);
	else if constexpr (std::is_same_v<T, double>)
		glClearBufferData(GL_ARRAY_BUFFER, GL_RGBA32UI, GL_RGBA, GL_UNSIGNED_INT, (unsigned int*)&vertex);

	glutil::glVertexAttribPtr<T>(0, 2, GL_FALSE, 2 * sizeof(T), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}
template <typename T>
LineStrip<T>::LineStrip(T* vertices, uint32_t count)
{
	this->count = count;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 2 * (count+1) * sizeof(T), vertices, GL_STREAM_DRAW);

	glutil::glVertexAttribPtr<T>(0, 2, GL_FALSE, 2 * sizeof(T), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}
template <typename T>
void LineStrip<T>::draw(const GLuint shader, const mat3<T>& viewMtx, const vec2<T> offset, const fvec3& color, const float linewidth)
{
	glUseProgram(shader);
	glLineWidth(linewidth);
	glutil::glUniformMatrix3v<T>(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (T*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glutil::glUniform2v<T>(glGetUniformLocation(shader, "offset"), 1, (T*)&offset);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINE_STRIP, 0, this->count + 1);
	glBindVertexArray(0);
}
template <typename T>
LineStrip<T>::~LineStrip()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}
template <typename T>
GLuint LineStrip<T>::getBuffer()
{
	return VBO;
}
template <typename T>
uint32_t LineStrip<T>::getCount()
{
	return count;
}