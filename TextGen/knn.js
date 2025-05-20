const fs = require("fs");
const path = require("path");

const datasets = {
    "tiny_shakespeare": new String(fs.readFileSync(path.join(__dirname, "/datasets/tiny_shakespeare.txt"))),
    "shrek": new String(fs.readFileSync(path.join(__dirname, "/datasets/shrek.txt"))),
    "bee_movie": new String(fs.readFileSync(path.join(__dirname, "/datasets/bee_movie.txt")))
}

const dataset = datasets["tiny_shakespeare"];

const maxRetries = 100;

function sampleIndex(probabilities) {
    const rand = Math.random();
    let acc = 0;
    for (let i = 0; i < probabilities.length; i++) {
        acc += probabilities[i];
        if (rand < acc) return i;
    }
    return probabilities.length - 1;
}

function softmaxSampling(obj, numSamples, temperature = 1.0) {
    const entries = Object.entries(obj);

    const scores = entries.map(([key, value]) => Math.exp(value / temperature));
    const sum = scores.reduce((a, b) => a + b, 0);
    const probabilities = scores.map(score => score / sum);

    let attempts = 0;

    const selected = new Set();
    while (attempts < maxRetries && selected.size < numSamples && selected.size < entries.length) {
        const index = sampleIndex(probabilities);
        selected.add(entries[index][0]);
        attempts++;
    }

    const result = {};
    for (const key of selected) {
        result[key] = obj[key];
    }

    return result;
}

class Model {
    constructor(data) {
        this.trainingData = data;

        this.uniqueWords = [];
        this.uniqueWordsIndex = [];
        this.data = {};

        this.generateWordTable();
    }

    generateWordTable() {
        this.splitData = this.trainingData.split(/\s+/);

        for(let i in this.splitData) {
            if(this.uniqueWords.indexOf(this.splitData[i]) < 0) {
                this.uniqueWords.push(this.splitData[i]);
                this.uniqueWordsIndex[this.splitData[i]] = [];
            }

            if(this.uniqueWordsIndex[this.splitData[i]]) this.uniqueWordsIndex[this.splitData[i]].push(i);
        }

        console.log("Successfully created unique word table");
    }

    trainModel() {
        const startTime = Date.now();

        for(let i in this.uniqueWords) {
            const proceedingWords = {};

            for (let j in this.uniqueWordsIndex[this.uniqueWords[i]]) {
                const index = parseInt(this.uniqueWordsIndex[this.uniqueWords[i]][j]);
                const proceedingWord = this.splitData[index + 1];

                if (proceedingWord === undefined) continue;

                if (!proceedingWords[proceedingWord]) proceedingWords[proceedingWord] = 1;
                else proceedingWords[proceedingWord]++;
            }

            this.data[this.uniqueWords[i]] = proceedingWords;
        }

        const endTime = (Date.now() - startTime) / 1000;

        console.log("Model successfully trained in " + endTime + " seconds");
    }

    generateText(word) {
        const candidates = this.data[word];
        
        if (!candidates || Object.keys(candidates).length === 0) {
            console.log(`No candidates found for word "${word}"`);
            return null;
        }

        //const firstStage = softmaxSampling(candidates, 100, 35);
        //const secondStage = softmaxSampling(firstStage, 50, 30);
        const thirdStage = softmaxSampling(candidates, 10, 50);
        const finalPick = softmaxSampling(thirdStage, 1, 1);

        return Object.keys(finalPick)[0];
    }

    generateSnippet(initial, length) {
        let generatingWord = initial;
        const generatedSnippet = [initial];

        for(let i = 0; i < length; i++) {
            const wordGenerated = this.generateText(generatingWord);
            
            generatedSnippet.push(wordGenerated);
            generatingWord = wordGenerated;
        }

        return generatedSnippet.join(" ");
    }
}

const model = new Model(dataset);

model.trainModel();

const text = model.generateSnippet("How", 250);
console.log(text);