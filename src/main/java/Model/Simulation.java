package Model;

import Model.Elements.*;
import View.ElementFrame;

import java.io.*;
import java.util.*;

public class Simulation extends Observable implements Serializable, Observer{

	private List<List<Element>> cells;  //This will hold a 2D array of all cells in the simulation
    private List<Agent> agents;
	private Set<Element> activeCells;   //This holds all cells in the simulation which are on fire or near fire
                                            //as these are the only ones who need to be updated
	private List<Simulation> states;    //This holds a list of previous states of the simulation if undo is set to true
                                            //otherwise it will only hold the first state for reset
    private int steps_taken = 0;

    private int width;
    private int height;
    private int step_time;
    private int step_size;
    private float wVecX;
    private float wVecY;
    private float windSpeed;
    private boolean undo_redo;

    // parameters related to agents
    private int nr_agents;
    private int energyAgents;
    private int fitness;

    private ParameterManager parameter_manager;
    private Generator generator;

	private boolean running;    //Boolean on whether the simulation it performing steps
    private boolean use_gui;

	private Random rand; //initializes RNG
    private long randomizer_seed = 0;

	public Simulation(boolean use_gui)
	{
        System.out.println("use_gui= " + use_gui );
	    this.use_gui = use_gui;
	    //Initialize these things
        Random seed_gen = new Random();
        randomizer_seed = seed_gen.nextLong();
        rand = new Random(randomizer_seed);
        states = new ArrayList<>();

        //Initialize the parameters to some default values and make them available for drawing
        create_parameters();

        parameter_manager = new ParameterManager(this);
        parameter_manager.addObserver(this);
        generator = new Generator(this);

        //Generate a new map to start on
        //generator.regenerate();
        generator.small();
        setChanged();
        notifyObservers(cells);
        //notifyObservers(agents);

        //This gathers the first set of cells to be active
		findActiveCells();
        //This adds the initial state to the states list
		//states.add((Simulation) deepCopy(this));
		if(!use_gui){
		    start();
        }
	}

    /**
     * Start is linked to the start button. It moves one step forward every Step_time in ms.
     * A negative time will make it perform steps backwards, but only if undo/redo is enabled.
     * The loop will stop when running is set to false by calling stop() or pressing the stop button.
     */
    public void start() {
	    running = true;
        while(running){
            if(step_time >=0){
                stepForward();
            }else{
                stepBack();
            }
            try {
                Thread.sleep(Math.abs((long) step_time));
            } catch (java.lang.InterruptedException e){
                System.out.println(e.getMessage());
            }

        }
    }

    /**
     * Pauses the simulation, linked to the stop button
     */
    public void stop(){
        running = false;
    }

    /**
     * Resets the simulation to the first state since the last regeneration. Linked to the reset button.
     */
    public void reset(){
	        stop();

            rand = new Random(randomizer_seed);
            states = new ArrayList<>();

            //generator.regenerate();
            generator.small();
            setChanged();
            notifyObservers(cells);
            notifyObservers(agents);

            findActiveCells();
    }

    /**
     * Clears all the cells and active cells and draws a new map.
     * Currently this is the tree_grid since we don't have a map generation.
     */
    public void regenerate() {
        stop();
        states.clear();
        activeCells.clear();

        //generator.regenerate();
        generator.small();
        setChanged();
        notifyObservers(cells);
        notifyObservers(agents);

        findActiveCells();
        states.add((Simulation) deepCopy(this));
    }


    /**
     *
     * Revert the simulation by one time step if undo/redo is enabled.
     * If there are no steps to take back anymore, the simulation is paused.
     * Linked to both the Step back button, as well as running the simulation with a negative step time.
     */
    public void stepBack(){
	    if(undo_redo == true){
	        for(int i = 0; i< step_size; i++) {
	            if(steps_taken > 0){
	                reset();
	                for(int j= 0; j < steps_taken; j++){
	                    stepForward();
                        setChanged();
                        notifyObservers(cells);
                    }

                }else{
	                running = false;
                }
                /*
                if (states.size() > 0) {
                    Simulation rewind = states.get(states.size() - 1);
                    states.remove(states.size() - 1);
                    this.cells = rewind.cells;
                    this.activeCells = rewind.activeCells;
                    setChanged();
                    notifyObservers(cells);
                } else {
                    running = false;
                }
                */
            }
        }
    }

    /**
     * Perform one step forward (and record the previous state if undo/redo is enabled).
     * The step forward is performed by updateEnvironment(), and the new state is sent to the GUI with notifyObservers()
     */
    public void stepForward(){
        for(int i = 0; i< step_size; i++) {
            if (undo_redo == true) {
                System.out.println("Adding undo_copy");
                states.add((Simulation) deepCopy(this));
                steps_taken++;
            }
            updateEnvironment();
        }
        setChanged();
        notifyObservers(cells);
        //notifyObservers(agents);
    }

    /**
     * This returns the 2D matrix of all cells currently in the simulation.
     * @return
     */
    public List<List<Element>> getAllCells() {
        return cells;
    }

	/**
	 *  Update the list of active cells. Apply the heat from the burning cell cell to all
	 *  of its neighbouring cells. If it ignites a neighbouring cell, add that cell to the
	 *  activeCells. If a burning cell runs out of fuel, remove it from the activeCells.
	 */
    public void updateEnvironment()
	{
		HashSet<Element> toRemove = new HashSet<>();
		HashSet<Element> toAdd = new HashSet<>();

		boolean onlyAgentsLeft = false;
		// should be if activeCells.size() == nr_agents
		if (true)
		{
			onlyAgentsLeft = true;
			for (Element agent : activeCells)
			{
				if (!agent.getType().equals("Agent"))
				{
					onlyAgentsLeft = false;
					break;
				}
			}
		}
		if (onlyAgentsLeft)
		{
			running = false;
			System.out.println(activeCells.size() + " VS " + nr_agents);
			for (Element e : activeCells)
			{
				System.out.println("Element " + e.getType() + ", at (" + e.getX() + "," + e.getY() + ")");
			}
			System.out.println("STOPPED");
		}

		// burningCell can also be an agent, they are counted as activeCells
		for (Element burningCell : activeCells)
		{
			String status = burningCell.timeStep();
            System.out.println("celltype: " + burningCell.getType() + " status: " + status);
            if (status.equals("Dead"))
            {
                toRemove.add(burningCell);
            }
            if (!burningCell.getType().equals("Agent")) {

			    if (status.equals("No Change"))
			    {
			    	HashSet<Element> neighbours = burningCell.getNeighbours(cells, agents);
                    System.out.println("size neigbours:" + neighbours.size());
                    for (Element e : neighbours){
                        System.out.println("activeCell has type: " + e.getType());
                    }
			    	for (Element neighbourCell : neighbours)
			    	{
			    		if (neighbourCell.isBurnable())
			    		{

                            neighbourCell.getHeatFrom(burningCell);
                            status = neighbourCell.timeStep();
                            if (status.equals("Ignited"))
                            {
                                toAdd.add(neighbourCell);
                            }
                        }

					}
				}
			}
		}
		activeCells.removeAll(toRemove);
		activeCells.addAll(toAdd);
	}

	/*
		Initializes activeCells by searching the entire map for burning cells
		and adding those and their neighbours
	 */
    private void findActiveCells()
	{
	    activeCells = new HashSet<>();
		for (int x = 0; x < width; x++)
		{
			for (int y = 0; y < height; y++)
			{
				Element cell = cells.get(x).get(y);
				if (cell.isBurning())
				{
					activeCells.add(cell);
				}
			}
		}

		for (int i = 0; i<nr_agents; i++){
            activeCells.add(agents.get(i));
        }
        for (Element e : activeCells){
            System.out.println("activeCell has type: " + e.getType() + " at temp: " + e.getTemperature());
        }
	}

    /**
     * This sets all tunable parameters to a default value, and adds it to the list of TextFields tuneable at runtime
     * Due to HashMap restrictions it only works with Strings and Floats, so you should initialize a value with 3f.
     * If you want to access the value of a parameter do parameters.get("Parameter name").floatValue()
     */
    public void create_parameters() {
        width = 11;
        height = 11;
        nr_agents = 3;
        energyAgents = 20;
        if(use_gui) {
            step_time = 100;
        }else{
            step_time = 0;
        }
        step_size = 1;
        undo_redo = false;
        wVecX = -1;
        wVecY = 0;
        windSpeed = 2;
    }


    /**
     * Return the parameters currently set, to be used by the parameter manager.
     * The values are defined in createParameters()
     * @return
     */
    public Map<String, Float> getParameters() {
        //TODO!
        //return parameters;
        Map<String, Float> return_map = new HashMap<>();
        return_map.put("Width", (float) width);
        return_map.put("Height", (float) height);
        return_map.put("Number of Agents", (float) nr_agents);
        return_map.put("Energy of Agents", (float) energyAgents);
        return_map.put("Step Size", (float) step_size);
        return_map.put("Step Time", (float) step_time);
        return_map.put("Undo/Redo", undo_redo ? 1f : 0f);
        return_map.put("Wind x", wVecX);
        return_map.put("Wind y", wVecY);
        return_map.put("Wind Speed", windSpeed);
        return return_map;
    }


    /**
     * This makes a full copy of any Serializable object, including it's children.
     * This is needed for being able to revert to previous states and circumventing Java's pass-by-reference
     * It's probably best to just leave this code as is unless you understand what is going on here.
     *
     */
    private static Object deepCopy(Object object) {
        try {
            ByteArrayOutputStream outputStream = new ByteArrayOutputStream();
            ObjectOutputStream outputStrm = new ObjectOutputStream(outputStream);
            outputStrm.writeObject(object);
            ByteArrayInputStream inputStream = new ByteArrayInputStream(outputStream.toByteArray());
            ObjectInputStream objInputStream = new ObjectInputStream(inputStream);
            return objInputStream.readObject();
        }
        catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * Simulation is observer to the parameterManager.
     * When the parameterManager changes something this update function is called,
     * with Object o a Map.Entry<String, Map.Entry<String, Float>>.
     * This way Object holds the recipient (here model, elsewhere i.e. Tree), the Parameter (i.e. Width) and the value
     * @param observable
     * @param o
     */
    @Override
    public void update(Observable observable, Object o) {
        if(o instanceof Map.Entry
            && ((Map.Entry) o).getKey() instanceof String
            && ((Map.Entry) o).getValue() instanceof Map.Entry
                && ((Map.Entry) o).getKey() == "Model"
                ){  //IF a Map.Entry<String, Map.Entry<String, Float>> and the first string is "Model"
            Float value = (Float) ((Map.Entry) ((Map.Entry) o).getValue()).getValue();
            switch( (String) ((Map.Entry) ((Map.Entry) o).getValue()).getKey() ){
                case "Width":
                    width = value.intValue();
                    break;
                case "Height":
                    height = value.intValue();
                    break;
                case "Step Time":
                    if(use_gui){
                        step_time = value.intValue();
                    }
                    break;
                case "Step Size":
                    step_size = value.intValue();
                    break;
                case "Undo/Redo":
                    undo_redo = value.intValue() == 1;
                    break;
                case "Number of Agents":
                    nr_agents = value.intValue();
                    break;
                case "Wind x":
                    wVecX = value;
                    break;
                case "Wind y":
                    wVecY = value;
                    break;
                case "Wind Speed":
                    windSpeed = value;
                    break;
                case "Energy of Agents":
                    energyAgents = value.intValue();
                default:
                    System.out.println("No action defined in Simulation.update for " + (String) ((Map.Entry) ((Map.Entry) o).getValue()).getKey());
            }
        }
    }

    /**
     * Needed to give controlPanel access to parameterManager
     * @return
     */
    public ParameterManager getParameter_manager(){
        return parameter_manager;
    }
    public Random getRand() {
        return rand;
    }

    public void setRand(Random rand) {
        this.rand = rand;
    }

    public int getRandX() {return rand.nextInt(width);}
    public int getRandY() {return rand.nextInt(height);}

    public List<Agent> getAgents() {
        return agents;
    }

    public int getNr_agents() {
        return nr_agents;
    }

    public void setNr_agents(int nr_agents) {
        this.nr_agents = nr_agents;
    }

    public void setCells(List<List<Element>> cells){ this.cells = cells; }

    public void setAgents(List<Agent> agents) {
        this.agents = agents;
    }

    public int getEnergyAgents() {
        return energyAgents;
    }

    public void setEnergyAgents(int energyAgents) {
        this.energyAgents = energyAgents;
    }

    public int getFitness() {
        return fitness;
    }

    public void setFitness(int fitness) {
        this.fitness = fitness;
    }


    /**
     * Debugging function
     */
    public void printCells() {
        for (int i =0; i<cells.get(0).size();i++){
            for (int j=0; j<cells.size(); j++){
                Element cell=cells.get(j).get(i);
                System.out.print(cell.getType() + " x:" + cell.getX() + " y:" + cell.getY() + " - ");
            }
            System.out.println();

        }
    }
}
