#include "Rect.h"
template class Rect<float>;
template class Rect<double>;

template <typename T>
Rect<T>::Rect() : VAO(0), VBO(0) {}
template <typename T>
Rect<T>::Rect(T vertices[8])
{
	glGenBuffers(1, &VBO);
	glGenVertexArrays(1, &VAO);

	glBindVertexArray(VAO);
	glBindBuffer(GL_ARRAY_BUFFER, VBO);
	glBufferData(GL_ARRAY_BUFFER, 8 * sizeof(T), vertices, GL_STREAM_DRAW);

	glutil::glVertexAttribPtr<T>(0, 2, GL_FALSE, 2 * sizeof(T), (void*)0);
	glEnableVertexAttribArray(0);
	glBindVertexArray(0);
}
template <typename T>
void Rect<T>::draw(const GLuint shader, const mat3<T>& viewMtx, const vec2<T>& offset, const fvec3& color, const float linewidth, const bool fill)
{
	glUseProgram(shader);
	glLineWidth(linewidth);
	glutil::glUniformMatrix3v<T>(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (T*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glutil::glUniform2v<T>(glGetUniformLocation(shader, "offset"), 1, (T*)&offset);
	glBindVertexArray(VAO);
	glDrawArrays(fill ? GL_TRIANGLE_FAN : GL_LINE_LOOP, 0, 4);
	glBindVertexArray(0);
}
template <typename T>
Rect<T>::~Rect()
{
	glDeleteBuffers(1, &VBO);
	glDeleteVertexArrays(1, &VAO);
}