package com.kraydel.edgedetection;


import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.util.ArrayList;
import java.util.List;

import static org.opencv.core.Core.*;
import static org.opencv.core.CvType.CV_8UC1;

public class EdgeDetectionActivity extends Activity
        implements CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                    openCvCameraView.enableView();
                    break;
                default:
                    super.onManagerConnected(status);
                    break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        openCvCameraView = new JavaCameraView(this, -1);
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
    }

    @Override
    public void onCameraViewStopped() {
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        Mat cannyFrame = new Mat(aInputFrame.size(), aInputFrame.type());
        Imgproc.cvtColor(aInputFrame, cannyFrame, Imgproc.COLOR_RGB2GRAY);
        Imgproc.Canny(cannyFrame, cannyFrame, 50, 150, 3, true);

        ArrayList<MatOfPoint> contours = new ArrayList<>();
        contours.clear();
        Imgproc.findContours(cannyFrame, contours, new Mat(), Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);
        for(MatOfPoint c: contours)
        {
            c.size();
            Rect boundingRect = Imgproc.boundingRect(c);
            Mat nothing = new Mat(aInputFrame, boundingRect);
            nothing.setTo(new Scalar(0, 0, 0));
        }
        Imgproc.drawContours(aInputFrame, contours, -1, new Scalar(255, 0, 0), 5);
        return aInputFrame;
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }
}