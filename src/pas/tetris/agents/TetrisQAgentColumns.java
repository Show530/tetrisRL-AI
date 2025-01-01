package src.pas.tetris.agents;


import java.util.Arrays;
// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;

// Column Heights and max height- not learning
public class TetrisQAgentColumns
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgentColumns(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        //final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        /*
        final int hiddenDim = 2 * numPixelsInImage;
        final int outDim = 1;
        */

        final int inputSize = 11;
        final int hiddenDim = 22;
        final int outDim = 1;


        Sequential qFunction = new Sequential();
        // qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Dense(inputSize, hiddenDim));
        // qFunction.add(new Tanh());
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
     * TODO:
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?

        Possible factors:
            Number of Holes
            Bumpiness
            Height
            Looking ahead at the next three pieces
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        // matrix to store our features
        Matrix features = Matrix.zeros(1, 11); 

        Matrix flattenedImage = null;
        try
        {
            // flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
            flattenedImage = game.getGrayscaleImage(potentialAction);
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }

        int[] columnHeights = new int[flattenedImage.getShape().getNumCols()];
        for (int i = 0; i < columnHeights.length; i++) {
            columnHeights[i] = -1;
        }


        for (int row = 0; row < flattenedImage.getShape().getNumRows(); row++) {
            for (int col = 0; col < flattenedImage.getShape().getNumCols(); col++) {
                // cell is occupied
                if(flattenedImage.get(row, col) == 1.0 || flattenedImage.get(row, col) == 0.5) {
                    if (columnHeights[col] == -1) {
                        columnHeights[col] = row;
                    }
                }

            }
        }

        double maxHeight = 0;
        // calculate column heights for matrix
        for (int i = 0; i < Board.NUM_COLS; i++) {
           if (columnHeights[i] == -1) {
                features.set(0, i, 0);
           }
           else {
                features.set(0, i, columnHeights[i]);
           }
           // calc max height
           if (columnHeights[i] > maxHeight) {
                maxHeight = columnHeights[i];
           }
            
        }

        features.set(0, Board.NUM_COLS, maxHeight);

        return features;
    }

    /**
     * TODO:
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        int currGame = (int)gameCounter.getCurrentGameIdx();

        double highestExploreProb = 1.0;
        double decayRate = 0.99 - (currGame * 0.01);
        double xyz = highestExploreProb - EXPLORATION_PROB;

        double currTurn = (double)gameCounter.getCurrentMoveIdx() * (xyz) / decayRate;

        // later on in the game we want to use knowledge we already know rather than taking a random move that may not be good
        double currentExplorationProb = Math.max(EXPLORATION_PROB, highestExploreProb - currTurn);

        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        return this.getRandom().nextDouble() <= currentExplorationProb;

    }

    /**
     * TODO:
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        //Mino bestMino = getBestActionAndQValue(game).getFirst();
        int numMinoPositions = game.getFinalMinoPositions().size();
        double[] qFunctions = new double[numMinoPositions];

        for(int i = 0; i < numMinoPositions; i++){
            Matrix cur = this.getQFunctionInput(game, game.getFinalMinoPositions().get(i));
            // gets the Q function
            try {
                qFunctions[i] = Math.exp(this.initQFunction().forward(cur).get(0, 0));
            }
            catch (Exception e) {
                e.printStackTrace();
                System.exit(-1);
            }
        }
        // get middle not min!!!
        // sort list, take
        //int minValPos = 0;
        //double minVal = Double.POSITIVE_INFINITY;

        // sum of all the q values
        double qSum = 0.0;
        for (int i = 0; i < numMinoPositions; i++) {
            qSum += qFunctions[i];
        }

        // normalizing
        double[] outcome = new double[numMinoPositions];
        double[] outcomesToSort = new double[numMinoPositions];
        for (int i = 0; i < numMinoPositions; i++) {
            outcome[i] = qFunctions[i] / qSum;
            outcomesToSort[i] = qFunctions[i] / qSum;
        }
        Arrays.sort(outcomesToSort);

        double med = outcomesToSort[numMinoPositions / 2];
        int indexOfMed = -1;
        for (int i = 0; i < numMinoPositions; i++) {
            if (med == outcome[i]) {
               indexOfMed = i;
            }
        }

        return game.getFinalMinoPositions().get(indexOfMed);
        
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * TODO:
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        Board board = game.getBoard();
        
        double reward = 15.0;
        double holePenalty = 0.0;
        double maxHeight = -1.0;

        //boolean isBoardEmpty = true;
        //int completedRows = 0;

        int[] columnHeights = new int[Board.NUM_COLS];
        for (int i = 0; i < Board.NUM_COLS; i++) {
            columnHeights[i] = -1;
           
        }

        for (int col = 0; col < Board.NUM_COLS; col++) {
            //boolean lineCleared = true; 
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                 
                if (board.isCoordinateOccupied(col, row)) {
                    //isBoardEmpty = false;
                    columnHeights[col] = Board.NUM_ROWS - row; 
                }
                else {
                    // Increment hole penalty for each hole
                    holePenalty += 1.0; 
                   
                    }

                }
            }
        
        
        for (int i = 0; i < Board.NUM_COLS; i++) {
            
            if (columnHeights[i] > maxHeight) {
                maxHeight = columnHeights[i];
            } 

        }
        

        int turnScore = game.getScoreThisTurn();

        
        reward -= maxHeight * 10.0;
        reward -= holePenalty * 30.0;
        reward += turnScore * 600.0;
        return reward;
    }
}
