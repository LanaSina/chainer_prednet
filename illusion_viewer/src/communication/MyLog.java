package communication;


/**
 * This class is used to log identified messages into the console.
 * Set shouldLog to false to stop logging.
 * @author lana
 *
 */
public class MyLog {

	/** name under which messages will be logged*/
	String name;
	/** set to false to temporarily silence the logger.*/
	boolean speak;
	
	/**
	 * Constructor
	 * @param n the name that should be used when logging.
	 * @param s set true is messages should be printed, false otherwise.
	 */
	public MyLog(String n, boolean s){
		name = n;
		speak = s;
	}
	
	/**
	 * Use this function to decide when the logger should log (default: true).
	 * @param s set true is messages should be printed, false otherwise.
	 */
	public void setSpeak(boolean s){
		speak = s;
	}
	
	/**
	 * Prints the name of the log object and a message.
	 * @param s the message to be printed.
	 */
	public void say(String s){
		if (speak && Constants.getShouldLog()){
			System.out.println(name + " says " + s);
		}
	}
	
	/**
	 * @param name of the logger. Avoid changing in runtime to prevent confusion.
	 * */
	public void setName(String name) {
		this.name = name;
	}

	public String getName() {
		return name;
	}

	public void say(double[] array) {
		String string = "";
		for (int i = 0; i < array.length; i++) {
			string+= " " + array[i];
		}
		
		say(string);
	}
}
