#include "CyclicalLineStrip.h"


CyclicalLineStrip::CyclicalLineStrip(glm::vec2 vertex, uint32_t count, glm::vec3 color) : LineStrip(vertex, count + 1, color)
{
	this->head = 0;
}
void CyclicalLineStrip::draw(GLuint shader, glm::mat3 viewMtx)
{
	glUseProgram(shader);
	glUniformMatrix3fv(glGetUniformLocation(shader, "viewMtx"), 1, GL_FALSE, (float*)&viewMtx);
	glUniform3fv(glGetUniformLocation(shader, "DiffuseColor"), 1, (float*)&color);
	glBindVertexArray(VAO);

	// [<head> <1> <2> <3> <4> <5> <6> <head>]	=> (head,N-1)
	// [<6> <head> <1> <2> <3> <4> <5> <6>]		=> (head,N-1)
	// [<4> <5> <6> <head> <1> <2> <3> <4>]		=> (0,head) + (head,N-head)
	if (head > 1)
	{
		glDrawArrays(GL_LINE_STRIP, 0, head);
		glDrawArrays(GL_LINE_STRIP, head, this->count - head);
	}
	else
		glDrawArrays(GL_LINE_STRIP, head, this->count - 1);
}
CyclicalLineStrip::~CyclicalLineStrip()
{ }
GLuint CyclicalLineStrip::getBuffer()
{
	return VBO;
}
void CyclicalLineStrip::finish()
{

}