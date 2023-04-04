#include "VideoEncoder.h"
#include <iostream>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/avutil.h>
#include <libavutil/time.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

//Modified from https://stackoverflow.com/a/59559256

void VideoEncoder::pushFrame(uint8_t* data)
{
    if (!this->initialized)
        return;

    int err;
    if (!videoFrame) {
        videoFrame = av_frame_alloc();
        videoFrame->format = AV_PIX_FMT_YUV420P;
        videoFrame->width = cctx->width;
        videoFrame->height = cctx->height;
        if ((err = av_frame_get_buffer(videoFrame, 32)) < 0) {
            std::cout << "Failed to allocate picture" << err << std::endl;
            return;
        }
    }
    if (!swsCtx) {
        swsCtx = sws_getContext(cctx->width, cctx->height, AV_PIX_FMT_RGB24, cctx->width, cctx->height, AV_PIX_FMT_YUV420P, SWS_BICUBIC, 0, 0, 0);
    }
    int inLinesize[1] = { 3 * cctx->width };
    // From RGB to YUV
    sws_scale(swsCtx, (const uint8_t* const*)&data, inLinesize, 0, cctx->height, videoFrame->data, videoFrame->linesize);
    videoFrame->pts = (1.0 / fps) * 91530 * (frameCounter++); // Magic number?
    if ((err = avcodec_send_frame(cctx, videoFrame)) < 0) {
        std::cout << "Failed to send frame" << err << std::endl;
        return;
    }
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;
    pkt.flags |= AV_PKT_FLAG_KEY;
    if (avcodec_receive_packet(cctx, &pkt) == 0) {
        av_interleaved_write_frame(ofctx, &pkt);
        av_packet_unref(&pkt);
    }
}

void VideoEncoder::close() {
    //DELAYED FRAMES
    AVPacket pkt;
    av_init_packet(&pkt);
    pkt.data = NULL;
    pkt.size = 0;

    for (;;) {
        avcodec_send_frame(cctx, NULL);
        if (avcodec_receive_packet(cctx, &pkt) == 0) {
            av_interleaved_write_frame(ofctx, &pkt);
            av_packet_unref(&pkt);
        }
        else {
            break;
        }
    }

    av_write_trailer(ofctx);
    if (!(oformat->flags & AVFMT_NOFILE)) {
        int err = avio_close(ofctx->pb);
        if (err < 0) {
            std::cout << "Failed to close file" << err << std::endl;
        }
    }
}

VideoEncoder::~VideoEncoder() {
    if (videoFrame) {
        av_frame_free(&videoFrame);
    }
    if (cctx) {
        avcodec_free_context(&cctx);
    }
    if (ofctx) {
        avformat_free_context(ofctx);
    }
    if (swsCtx) {
        sws_freeContext(swsCtx);
    }
}

bool VideoEncoder::initialize()
{
    av_log_set_level(AV_LOG_QUIET);

    oformat = av_guess_format(nullptr, filename, nullptr);
    if (!oformat)
    {
        std::cout << "can't create output format" << std::endl;
        return false;
    }
    //oformat->video_codec = AV_CODEC_ID_H265;

    if (avformat_alloc_output_context2(&ofctx, oformat, nullptr, filename))
    {
        std::cout << "can't create output context" << std::endl;
        return false;
    }

    codec = avcodec_find_encoder(oformat->video_codec);
    if (!codec)
    {
        std::cout << "can't create codec" << std::endl;
        return false;
    }

    stream = avformat_new_stream(ofctx, codec);
    if (!stream)
    {
        std::cout << "can't find format" << std::endl;
        return false;
    }

    cctx = avcodec_alloc_context3(codec);
    if (!cctx)
    {
        std::cout << "can't create codec context" << std::endl;
        return false;
    }


    stream->codecpar->codec_id = oformat->video_codec;
    stream->codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
    stream->codecpar->width = width;
    stream->codecpar->height = height;
    stream->codecpar->format = AV_PIX_FMT_YUV420P;
    stream->codecpar->bit_rate = bitrate;
    avcodec_parameters_to_context(cctx, stream->codecpar);
    cctx->time_base = { 1 , fps };
    cctx->max_b_frames = 2;
    cctx->gop_size = 12;
    cctx->framerate = { fps , 1 };

    
    if (stream->codecpar->codec_id == AV_CODEC_ID_H264) {
        av_opt_set(cctx, "preset", "medium", 0);
    }/*
    else if (stream->codecpar->codec_id == AV_CODEC_ID_H265)
    {
        av_opt_set(cctx, "preset", "medium", 0);
    }*/

    avcodec_parameters_from_context(stream->codecpar, cctx);

    int err;
    if ((err = avcodec_open2(cctx, codec, NULL)) < 0) {
        std::cout << "Failed to open codec" << err << std::endl;
        return false;
    }

    if (!(oformat->flags & AVFMT_NOFILE)) {
        if ((err = avio_open(&ofctx->pb, filename, AVIO_FLAG_WRITE)) < 0) {
            std::cout << "Failed to open file" << err << std::endl;
            return false;
        }
    }

    if ((err = avformat_write_header(ofctx, NULL)) < 0) {
        std::cout << "Failed to write header" << err << std::endl;
        return false;
    }

    av_dump_format(ofctx, 0, filename, 1);

    this->initialized = true;
    return true;
}

VideoEncoder::VideoEncoder(const char* filename, int width, int height, int fps, int bitrate)
{
    this->initialized = false;
    this->width = 2*(width/2);
    this->height = 2*(height/2);
    this->fps = fps;
    this->bitrate = bitrate;
    this->filename = filename;
}