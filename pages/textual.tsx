import { oneHot } from "@tensorflow/tfjs";

const Textual = () => {
    const words = ["paris", "london","berlin"];

    const encode_word = (word:string) =>{

        const word_index = words.indexOf(word);
        const Is_word_available = !!(words.indexOf(word) !== -1);
        if(!Is_word_available){throw new Error("non-existing word")};

        const encoded = oneHot(word_index,words.length);
        return encoded;
    }

    const output = encode_word("london");
    console.log(output.arraySync())

    return ( 
        <>
            <h1>Machine learning with Textual Data</h1>
        </>
     );
}
 
export default Textual;