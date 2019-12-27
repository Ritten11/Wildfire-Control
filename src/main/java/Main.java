
import Learning.BURLAP;
import Learning.CoSyNe.CircleSyNe;
import Learning.CoSyNe.HRL.ActionLearner;
import Learning.CoSyNe.SubGoalLearning;
import Learning.CoSyNe.SubSyne;
import Learning.DeepQ.DeepQLearner;
import Learning.GA;
import Learning.HumanController;
import Model.Simulation;
import View.MainFrame;
import org.apache.commons.lang3.time.StopWatch;

public class Main {
	public static void main(String[] args) {

		StopWatch watch = new StopWatch();
		watch.start();

		boolean use_gui;
		if (args.length == 1 && args[0].equals("no_gui")) {
			System.out.println("NO GUI!");
			use_gui = false;
			new Simulation(use_gui, 1).start();
		} else if (args.length == 1 && args[0].equals("cosyne_gui")) {
			System.out.println("CoSyNe gui");
			//new GA();
			new CircleSyNe();
		} else if (args.length == 1 && args[0].equals("GA")) {
			System.out.println("GA");
			new GA();
		}  else if (args.length == 1 && args[0].equals("BURLAP")) {
			BURLAP test = new BURLAP();
			test.example();
		}else if (args.length == 1 && args[0].equals("HRL")){
			new ActionLearner();
		} else if (args.length > 0 && args[0].equals("CQL")){
            new DeepQLearner(1);
        } else if (args.length > 0 && args[0].equals("CoSyNe_SubGoals")){
            new SubGoalLearning(1);
        }else if (args.length == 1 && args[0].equals("human")) {
			HumanController hc = new HumanController();
			Simulation s = new Simulation(hc, 1);
			hc.setModel(s);
			MainFrame f = new MainFrame(s);
			f.simulationPanel.addKeyListener(hc);
			hc.simulationPanel = f.simulationPanel;
		} else if(args.length > 1 && Integer.parseInt(args[1]) > 0 && Integer.parseInt(args[1]) <= 8) {
			if (args[0].equals("CQL")){
				System.out.println("CQL");
				new DeepQLearner(Integer.parseInt(args[1]));

			}else if (args[0].equals("sub")){
				new SubGoalLearning(Integer.parseInt(args[1]));
			}
			else if (args[0].equals("CoSyNe_SubGoals")){
				new SubGoalLearning(Integer.parseInt(args[1]));
			}
		}
		 else {
			System.out.println("Oops, something went wrong... Please check whether you entered the arguments in the right order (STRING, INT, STRING)");

			/*// Roel:
			Features features = new Features();
			features.downSampledFuelMap(model, 3, 3,1);
			*/
		}
	}



}


