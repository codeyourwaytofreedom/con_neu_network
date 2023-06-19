import { oneHot } from "@tensorflow/tfjs";
import * as stopwords from 'stopword';


const Textual = () => {
    const sports = ["boxing", "tennis","rowing"];
    const economics = ["money", "currency","cash"];

    const test_sentence = "China prides itself on firm, “unswerving” leadership and stable economic growth. \
    That should make its fortunes easy to predict. But in recent months, the world’s second-biggest economy \
    has been full of surprises, wrong-footing seasoned China-watchers and savvy investors alike."

    const sterile = stopwords.removeStopwords(test_sentence.split(' ')).filter(word => word.trim() !== '');
    console.log(sterile);




    const encode_word = (word:string, vocab_group:string[]) =>{

        const word_index = vocab_group.indexOf(word);
        const Is_word_available = !!(vocab_group.indexOf(word) !== -1);
        if(!Is_word_available){throw new Error("non-existing word")};

        const encoded = oneHot(word_index,vocab_group.length);
        return encoded;
    }

    const output = encode_word("boxing",sports);
    console.log(output.arraySync());

    return ( 
        <>
            <h1>Machine learning with Textual Data</h1>
        </>
     );
}
 
export default Textual;