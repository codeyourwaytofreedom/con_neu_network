import { use, useEffect, useRef, useState } from "react";
import hp from "../styles/Hp.module.css";
import * as tf from "@tensorflow/tfjs";
import { GraphModel } from "@tensorflow/tfjs";


const Hp = () => {
    const VIDEO = useRef<HTMLVideoElement>(null);
    const ENABLE_CAM_BUTTON = useRef<HTMLButtonElement>(null);
    const TRAIN_BUTTON = useRef<HTMLButtonElement>(null);
    const MN_INPUT_WID  = 224;
    const MN_INPUT_HEI = 224;
    const STOP_DATA_GATHER = -1;

    const [mobilnet, setMobileNet] = useState<GraphModel>();
    const [gatherDataState,setGatherDataState] = useState(STOP_DATA_GATHER);
    const [videoPlaying,setVidPlaying] = useState(false);
    const [trainingDataInputs,setTrainingDataInputs] = useState<any>([]);
    const [trainingDataOutputs,setTrainingDataOutputs] = useState<any>([]);

    const [example_count,setExample_count] = useState<number>();
    let class_names = [];

    let predict = false;
    
    const [animation_frame, setAnimFrame] = useState<any>();

    const loadMobilNetFeatureModel = async () =>{
        const URL = "https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_small_100_224/feature_vector/5/default/1";
        const mb = await tf.loadGraphModel(URL, {fromTFHub:true});
        if(!mobilnet){
            setMobileNet(mb);
        }
        tf.tidy(function(){
            let answer:any =  mb?.predict(tf.zeros([1,MN_INPUT_HEI,MN_INPUT_WID,3]));
            console.log(answer.shape)
        })
    }
    
    loadMobilNetFeatureModel();

    const hasGetUserMedia = () =>{
        return !! (navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
    }

    const enableCam = () => {
        if(hasGetUserMedia()){
            const constraints = {
                video:true,
                width:640,
                height:480
            }
            navigator.mediaDevices.getUserMedia(constraints).then((stream)=>{
                VIDEO.current!.srcObject = stream;

            })
        }
    }

      
    const dataGatherLoop = () => {
        if(videoPlaying && gatherDataState !== STOP_DATA_GATHER){
            let image_features = tf.tidy(()=>{
                let videoFrameasTensor = tf.browser.fromPixels(VIDEO.current!);
                let resized_Tensor_Frame = tf.image.resizeBilinear(videoFrameasTensor,[MN_INPUT_HEI,MN_INPUT_WID],true);
                let normalizedTensorFrame = resized_Tensor_Frame.div(255);
                return mobilnet?.predict(normalizedTensorFrame.expandDims()).squeeze(); //could be an issue here. double check
            });
            
            console.log("here");

            setTrainingDataInputs([...trainingDataInputs,image_features]);
            setTrainingDataOutputs([...trainingDataOutputs, gatherDataState]);

            console.log(image_features)


            let animID = window.requestAnimationFrame(dataGatherLoop);
            setAnimFrame(animID);
        }
    }

    const gatherDataforClass = (event: React.MouseEvent<HTMLButtonElement, MouseEvent>) => {
        const classNumber = parseInt(event.currentTarget.getAttribute('data-1hot')!);
        setGatherDataState(classNumber);       
    };

    useEffect(()=>{
        if(videoPlaying && gatherDataState !== STOP_DATA_GATHER){
            dataGatherLoop();
        }
        else{
                window.cancelAnimationFrame(animation_frame);
        }
    },[gatherDataState])

/* 
    useEffect(()=>{
        let model = tf.sequential();
        model.add(tf.layers.dense({inputShape:[1024],units:128,activation:"relu"}));
        model.add(tf.layers.dense({units:2,activation:"softmax"}));
        model.summary();

        model.compile({
            optimizer:"adam",
            loss:(class_names.length === 2) ? "binaryCrossentropy" : "categoricalCrossentropy",
            metrics:["accuracy"]
        });
    },[]) */

    return ( 
        <>
            <div className={hp.frame}>
                <h1>Hello CNN</h1>
                <video autoPlay ref={VIDEO} onLoadedData={()=> setVidPlaying(true)}/>
                <button ref={ENABLE_CAM_BUTTON} onClick={enableCam}>Open Cam</button>
                <button data-1hot = {0} onMouseDown={gatherDataforClass} onMouseUp={()=> setGatherDataState(-1)} onMouseLeave={()=> setGatherDataState(-1)}>GAther1</button>

                <button ref={TRAIN_BUTTON}>XXX</button>
                <button>Train & Predict</button>
                <h1>{example_count && example_count}</h1>

                <h1>{gatherDataState}</h1>
            </div>
        </>
     );
}
 
export default Hp;