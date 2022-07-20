package com.example.imagepro;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.app.Activity;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.Window;
import android.view.WindowManager;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import java.io.IOException;

public class CameraActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2{
    private static final String TAG="MainActivity";

    private Mat mRgba;   //Mat is n-dimensional dense array class.
                        //for which we are creating a object named mRgba.
    private Mat mGray;
    private CameraBridgeViewBase mOpenCvCameraView; //openCv
    //basically it is a class used for interaction between camera and OpenCV library.And to apply this client must implement the CvCameraViewListener
    private face_Recognition face_Recognition; //points at the class face_Recognition.

    private BaseLoaderCallback mLoaderCallback =new BaseLoaderCallback(this) {
        //BaseLoaderCallBack is a class
        @Override
        public void onManagerConnected(int status) {   //here status is ,status of initialization
            switch (status){
                case LoaderCallbackInterface
                        .SUCCESS:{
                    Log.i(TAG,"OpenCv Is loaded");      //it means opencv is successfully connected.
                    mOpenCvCameraView.enableView();
                }
                default:
                {
                    //callback method after the OpenCV initialization.
                    super.onManagerConnected(status); //if we have an overridden method we must call the parent class
                    //if we want to call the child class then we use 'this' keyword
                    //if we want to call the parent class then we use 'super' keyword

                }
                break;
            }
        }
    };

    public CameraActivity(){
        Log.i(TAG,"Instantiated new "+this.getClass());
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        int MY_PERMISSIONS_REQUEST_CAMERA=0;

        // if camera permission is not given it will ask for it on device
        if (ContextCompat.checkSelfPermission(CameraActivity.this, Manifest.permission.CAMERA)
                == PackageManager.PERMISSION_DENIED){
            ActivityCompat.requestPermissions(CameraActivity.this, new String[] {Manifest.permission.CAMERA}, MY_PERMISSIONS_REQUEST_CAMERA);
        }

        setContentView(R.layout.activity_camera);

        mOpenCvCameraView=(CameraBridgeViewBase) findViewById(R.id.frame_Surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

        try{
            int inputSize = 96;
            face_Recognition = new face_Recognition(getAssets(),
                    CameraActivity.this,
                    "model.tflite",
                    inputSize);
        }
        catch (IOException e){
            e.printStackTrace();
            Log.d("CameraActivity", "Model is not Loaded");
        }

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (OpenCVLoader.initDebug()){  //initDebug- it basically loads and initialize for current package.
            //if load success
            Log.d(TAG,"Opencv initialization is done");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
        else{
            //if not loaded
            Log.d(TAG,"Opencv is not loaded. try again");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0,this,mLoaderCallback);
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }
    }

    public void onDestroy(){
        super.onDestroy();
        if(mOpenCvCameraView !=null){
            mOpenCvCameraView.disableView();
        }

    }

    public void onCameraViewStarted(int width ,int height){
        mRgba=new Mat(height,width, CvType.CV_8UC4);
        mGray =new Mat(height,width,CvType.CV_8UC1);
    }
    public void onCameraViewStopped(){
        mRgba.release();
    }


    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame){ //it is a  callback function and it is called on retrieving frame from camera.
        //it is basically a object for the class CvCameraViewFrame.
        //and this callback is done only on the onCameraFrame because outside this function it is unpredictable.

        mRgba=inputFrame.rgba();         //it has two methods rgba() and gray().
        mGray=inputFrame.gray();
        mRgba = face_Recognition.recognizeImage(mRgba);

        return mRgba;

    }

}