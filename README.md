# What is this?

This is a text to handwriting python script. It takes any text, "hello world," and transforms it into a handwritten image of that text. It can do all english digts and letters. It uses the EMNIST dataset which is included in the dataset folder.

# Configuration

There are few settings that you can play with:

- letter spacing
- word spacing
- line spacing
- line width
- apply any post-fx function per letter written

# Info

The handwriting of each letter is different than the last. The EMNIST dataset has no organization of handwriting styles.

I designed this for training a ML model so there is a significant amount of random variance introduced at each stage.
