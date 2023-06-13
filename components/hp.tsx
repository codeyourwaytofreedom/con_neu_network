import { useEffect, useRef, useState } from "react";
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
    let gatherDataState = STOP_DATA_GATHER;
    let videoPlaying = false;
    let trainingDataInputs = [];
    let trainingDataOutputs = [];
    let example_count = [];
    let predict = false;
    let class_names = [];


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
    },[])

    return ( 
        <>
            <div className={hp.frame}>
                <h1>Hello CNN</h1>
                <video autoPlay ref={VIDEO}/>
                <button data-1hot = {0} ref={ENABLE_CAM_BUTTON}>Open Cam</button>
                <button data-1hot = {1} ref={TRAIN_BUTTON}>Open Cam</button>
                <button>Train & Predict</button>
            </div>
        </>
     );
}
 
export default Hp;