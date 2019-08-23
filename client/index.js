var HttpProxyAgent = require('http-proxy-agent');

let proxy = 'http://194.138.0.25:9400';
let opts = { agent: new HttpProxyAgent(proxy) };

var socket = require('socket.io-client').connect('http://95.177.164.90:5000/test', opts);
console.log("connecting...");
// var socket = require('socket.io-client')('http://localhost:5000/test');
const say = require('say');
var spoken_names = {};
var greetings = ['How do you do', 'Hello', 'Hi', 'Hai', 'Hey', 'How have you been', 'How are you',
    'How is it going', 'Salam alikom ', 'Esh loonak ya', 'Ahlaaaan']

socket.on('connect', function () {
    console.log("connected");
});
socket.on('event', function (data) {
    console.log("event", data);
});
socket.on('got_face', function (data) {
    console.log("event", data);
    if (!data.data) return;
    let current_person_list = [];
    if (data.data.length > 2) {
        speak("Welcome to MindSphere");
        return;
    }
    for (var i = 0, len = data.data.length; i < len; i++) {
        var element = data.data[i];
        if (!element || element === "Unknown") return;

        if (!spoken_names.hasOwnProperty(element)) {
            spoken_names[element] = new Date();
            // speak("Welcome  " + element);
            current_person_list.push("Welcome  " + element)
        }
        else {
            var timeNow = new Date();
            var seconds = (timeNow.getTime() - spoken_names[element].getTime()) / 1000;

            if (seconds >= 10) {
                spoken_names[element] = new Date();
                // speak(greetings[Math.floor(Math.random() * greetings.length)] + element);
                current_person_list.push(greetings[Math.floor(Math.random() * greetings.length)] + " " + element)
            }
        }
    }
    speak(current_person_list.join(" , "))
});
socket.on('disconnect', function () {
    console.log("disconnect");
});

function speak(text) {
    say.stop()
    say.speak(text)
}
