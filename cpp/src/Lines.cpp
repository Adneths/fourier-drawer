#include "Lines.h"
template class Lines<float>;
template class Lines<double>;

template <typename T>
Lines<T>::Lines(vec2<T> vertex, uint32_t count, bool timestamped, GLuint fillShader) : timestamped(timestamped), fillShader(fillShader)
{
	this->count = count;

	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, (timestamped ? 6ull : 4ull) * count * sizeof(T), nullptr, GL_DYNAMIC_DRAW);
	fill(vec3<T>(vertex, 0.0));
	// if constexpr (std::is_same_v<T, float>)
	// 	glClearBufferData(GL_ARRAY_BUFFER, timestamped ? GL_RGB32F : GL_RG32F, GL_RGBA, GL_FLOAT, &vec3<T>(vertex, 0.0));
	// else if constexpr (std::is_same_v<T, double>)
	// 	glClearBufferData(GL_ARRAY_BUFFER, timestamped ? 0/*???*/ : GL_RGBA32UI, GL_RGBA, GL_UNSIGNED_INT, (unsigned int*)&vec3<T>(vertex, 0.0));
	
	glutil::glVertexAttribPtr<T>(0, 2, GL_FALSE, (timestamped ? 3ull : 2ull) * sizeof(T), (void*)0);
	glEnableVertexAttribArray(0);
	if (timestamped)
	{
		glutil::glVertexAttribPtr<T>(1, 1, GL_FALSE, 3ull * sizeof(T), (void*)(2 * sizeof(T)));
		glEnableVertexAttribArray(1);
	}
	glBindVertexArray(0);
}
template <typename T>
void Lines<T>::draw(const GLuint shader, const mat3<T>&viewMtx, const vec2<T>&offset, const fvec3& color, const float linewidth)
{
	glUseProgram(shader);
	glLineWidth(linewidth);
	glutil::glUniformMatrix3v<T>(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (T*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glutil::glUniform2v<T>(glGetUniformLocation(shader, "offset"), 1, (T*)&offset);
	glBindVertexArray(VAO);
	glDrawArrays(GL_LINES, 0, this->count * 2);
	glBindVertexArray(0);
}
template <typename T>
Lines<T>::~Lines()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}
template <typename T>
void Lines<T>::fill(vec3<T> val) {
	glutil::fill_buffer<T>(fillShader, VBO, (T*)&val, (timestamped ? 3ull : 2ull), 0, (timestamped ? 6ull : 4ull) * count);
}
template <typename T>
void Lines<T>::fill(T val[3]) {
	fill(vec3<T>(val[0], val[1], val[2]));
}
template <typename T>
GLuint Lines<T>::getBuffer()
{
	return VBO;
}
template <typename T>
uint32_t Lines<T>::getCount()
{
	return count;
}