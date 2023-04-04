#pragma once

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

class VideoEncoder {
private:
	AVFrame* videoFrame = nullptr;
	AVCodecContext* cctx = nullptr;
	const AVCodec* codec = nullptr;
	SwsContext* swsCtx = nullptr;
	int frameCounter = 0;
	AVFormatContext* ofctx = nullptr;
	const AVOutputFormat* oformat = nullptr;
	AVStream* stream = nullptr;

	int fps;
	int width;
	int height;
	int bitrate;
	bool initialized;
	const char* filename;
public:
	VideoEncoder(const char* filename, int width, int height, int fps, int bitrate = 1000000);
	~VideoEncoder();
	bool initialize();
	void pushFrame(uint8_t* data);
	void close();
};