import { use, useEffect, useRef, useState } from "react";
import hp from "../styles/Hp.module.css";
import * as tf from "@tensorflow/tfjs";
import { GraphModel, model, Tensor } from "@tensorflow/tfjs";
import Image from "next/image";


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

    const [example_count,setExample_count] = useState<any[]>([
        {name:"Default",count:0},
        { name:"0", count:0}, { name:"1", count:0},{ name:"2", count:0},{ name:"3", count:0},
         { name:"4", count:0}, { name:"5", count:0} ,{ name:"6", count:0},{ name:"7", count:0},
        { name:"8", count:0},{ name:"9", count:0},{ name:"*", count:0},{ name:"+", count:0},
        { name:"-", count:0},{ name:"/", count:0},{ name:"=", count:0},{ name:"cln", count:0},

    ]);

    let class_names = ["default", "0","1","2","3","4","5", "6","7","8","9","*","+","-","/","=","cln"];

/*     const [example_count,setExample_count] = useState<any[]>([
        {name:"Default",count:0},
        { name:"0", count:0}, { name:"1", count:0},{ name:"2", count:0},{ name:"3", count:0},
         { name:"4", count:0}, { name:"5", count:0} ,{ name:"6", count:0},{ name:"7", count:0},
        { name:"8", count:0},{ name:"9", count:0},
    ]);

    let class_names = ["default", "0","1","2","3","4","5", "6","7","8","9"]; */

    const [predict, setPredict] = useState(false);

    const animationFrameIdRef = useRef<number>();
    const [model, setModel] = useState<any>();

    
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
    
    
    useEffect(()=>{
        loadMobilNetFeatureModel();
    },[]);



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
            
            setTrainingDataInputs([...trainingDataInputs,image_features]);
            setTrainingDataOutputs([...trainingDataOutputs, gatherDataState]);

            console.log(image_features);

            setExample_count(prevState => (
                prevState.map((item, index) => {
                  if (index === gatherDataState) {
                    return {
                      ...item,
                      count: item.count + 1
                    };
                  }
                  return item;
                })
              ));
            const animID = window.requestAnimationFrame(dataGatherLoop);
            animationFrameIdRef.current = animID;
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
            if (animationFrameIdRef.current) {
                window.cancelAnimationFrame(animationFrameIdRef.current);
              }
        }
        console.log(example_count)
    },[gatherDataState])

    const volume = class_names.length;
    const [aval, setAval] = useState(false);

    useEffect(()=>{
        const load_existing_model = async () =>{
            try{
                const existing_model = await tf.loadLayersModel('localstorage://whole');
                if(existing_model){
                    setModel(existing_model);
                    console.log("model avaiaible");
                    setAval(true)
                } else{
                    return
                }
            }
            catch{
                let modd = tf.sequential();
                modd.add(tf.layers.dense({inputShape:[1024],units:450,activation:"relu"}));
                modd.add(tf.layers.dense({units:volume,activation:"softmax"}));
                modd.summary();
        
                modd.compile({
                    optimizer:"adam",
                    loss:(class_names.length === volume) ? "binaryCrossentropy" : "categoricalCrossentropy",
                    metrics:["accuracy"]
                });
                setModel(modd);
            }
        }
        load_existing_model();
/*         let modd = tf.sequential();
        modd.add(tf.layers.dense({inputShape:[1024],units:450,activation:"relu"}));
        modd.add(tf.layers.dense({units:volume,activation:"softmax"}));
        modd.summary();

        modd.compile({
            optimizer:"adam",
            loss:(class_names.length === volume) ? "binaryCrossentropy" : "categoricalCrossentropy",
            metrics:["accuracy"]
        });
        setModel(modd); */
        
    },[])

    const trainAndPredict = async () =>{
        if(!aval && trainingDataInputs.length > 0){
            tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
            let outoutAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
            let oneHotOutputs = tf.oneHot(outoutAsTensor, class_names.length);
            let inputasTensor = tf.stack(trainingDataInputs);
    
            let results = await model.fit(inputasTensor, oneHotOutputs,{
                shuffle:true,
                batchSize:100,
                epochs:50,
                callbacks:{onEpochEnd: logProgress}
            })
    
            outoutAsTensor.dispose();
            oneHotOutputs.dispose();
            inputasTensor.dispose();
        }

        setPredict(true);
        predictLoop();
    }

    const logProgress = (ep:any, logs:any) =>{
        console.log(ep,logs)
    }

    const saveModel = async () => {
        const saveResult = await model.save('localstorage://numbers');
        console.log('Model saved:', saveResult);
      }

    const saveWhole_model = async () => {
        const saveResult = await model.save('localstorage://whole');
        console.log('Model saved:', saveResult);
      }

    const [result,setResult] = useState<any>();
    const _ops = ["-","+","/","*","cln"];
    
    const _95 = ["5"];
    const _85 = ["7"];
    const _80 = ["0","8","*","/","+","="];
    const _70 = ["1","2","9","6"];
    const _50 = ["3","4","-","cln"];





    const predictLoop = () => {
        if(predict){
            tf.tidy(()=>{
                let videoFrameAsTensor = tf.browser.fromPixels(VIDEO.current!).div(2555);
                let resized_Tensor_Frame = tf.image.resizeBilinear(videoFrameAsTensor as any,[MN_INPUT_HEI,MN_INPUT_WID],true);
                let image_features = mobilnet?.predict(resized_Tensor_Frame.expandDims());
                let prediction = model.predict(image_features).squeeze();
                let heighestIndex = prediction.argMax().arraySync();
                let predictionArray = prediction.arraySync();
                const selective_percentage = _95.includes(class_names[heighestIndex]) ? 95
                                            : _85.includes(class_names[heighestIndex]) ? 85
                                            : _80.includes(class_names[heighestIndex]) ? 80
                                            : _70.includes(class_names[heighestIndex]) ? 70
                                            : _50.includes(class_names[heighestIndex]) ? 50
                                            : 55;

                if(predictionArray[heighestIndex]*100 > selective_percentage){
                    setResult({
                    name:class_names[heighestIndex],
                    ratio:predictionArray[heighestIndex]*100
                });  
                }
            });
        }
        window.requestAnimationFrame(predictLoop);
    }

    const [last_number, setLast] = useState();
    useEffect(()=>{
        if(result){
            setLast(result.name);
        }
    },[result]);

    const [ins, setIns] = useState<any>([]);
    useEffect(()=>{
        if(last_number === "default"){
            return;
        }
        if(last_number !== "default"){
            setTimeout(() => {
                if(_ops.includes(last_number!) && _ops.includes(ins[ins.length-1])){
                    setIns(ins.map((e:any,i:any)=> i === ins.length-1 ? last_number : e));
                }
                if(_ops.includes(last_number!) && ins.length === 0){
                    return
                }
                else if(last_number === "="){
                    if(_ops.includes(ins[ins.length-1]) || ins.length === 0 || !ins.some((item:any) => _ops.includes(item))){
                        return;
                    }
                    else{
                        const united = ins.join("");
                        const operator = ins.find((e:any)=>_ops.includes(e));
                        const first = parseInt(united.split(operator)[0]); 
                        const second = parseInt(united.split(operator)[1]); 
                    
                        const rslt = operator === "*" ? first * second :
                                     operator === "/" ? first / second :
                                     operator === "+" ? first + second :
                                                          first - second;
                        setFinal(rslt)
                    }
                }
                else if (last_number === "cln"){
                    setIns([]);
                    setFinal(null)
                }
                else{
                    if(_ops.includes(last_number!) && ins.some((item:any) => _ops.includes(item))){
                        return
                    }
                    else{
                        setIns([...ins,last_number])
                    }
                    
                }
            }, 1500);
        }
    },[last_number]);

    const [final_output, setFinal] = useState<any>();
    const hide = useRef<HTMLInputElement>(null);
    const [display_controls, SetDisplay] = useState(false);

    return ( 
        <>
            <div className={hp.frame}>
                <div className={hp.frame_main}>
                    <input type="checkbox" ref={hide} onChange={()=>SetDisplay(hide.current!.checked)} />
                    <video autoPlay ref={VIDEO} onLoadedData={()=> setVidPlaying(true)}/>
                    <div className={hp.frame_main_board}>
                        <div className={hp.frame_main_board_motto}>
                            <Image src={"/tf.png"} alt={"tensorflow"} width={200} height={200}/>
                            <h1>Fun Math for Kids...</h1>
                        </div>
                        {
                            last_number && last_number !== "default" ? 
                            <div id={hp.last}>{last_number && last_number !== "default" && last_number}</div>
                            : null
                        }
                            <div className={hp.frame_main_board_result}>
                            {
                                ins.map((e:any,i:any)=>
                                    <span key={i}>{e}</span>
                                )
                            }
                            {final_output && '\u00A0' + "=" + '\u00A0' + final_output}
                            </div>                            
                    </div>
                </div>
                {
                    display_controls ? 
                    <>
                        <div className={hp.frame_controls}>
                            <button ref={ENABLE_CAM_BUTTON} onClick={enableCam}>Open Cam</button>
                            <button data-1hot = {0} data-name={"Group 1"} onMouseDown={gatherDataforClass} 
                                onMouseUp={()=> setGatherDataState(-1)}>Default</button>

                            {
                                class_names.slice(1,class_names.length).map((e,i)=>
                                <button data-1hot = {i+1} onMouseDown={gatherDataforClass} 
                                onMouseUp={()=> setGatherDataState(-1)}> {e} </button>
                                
                                )
                            }

                            
                            <button onClick={trainAndPredict}>Train & Predict</button>

                        </div>

                        <div className={hp.frame_checks}> 
                                {
                                    example_count.map((e,i)=>
                                    <h2 key={i}>{e.name}: {e.count}</h2>
                                    )
                                }

                            <br /> <h1>{result && result.name} - {result && result.ratio}</h1>
                        </div>

                        <button onClick={()=>setIns([])}>Clear board</button>
                        {/* <button onClick={saveModel}>Save numbers model</button> */}
                        <button onClick={saveWhole_model}>Save whole model</button>
                    </>
                    : null
                }


            </div>
        </>
     );
}
 
export default Hp;