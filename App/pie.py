# Display the appropriate risk percentage and no-risk percentage based on the prediction result
            if st.session_state['result'] == "High Risk":
                risk_percentage = st.session_state['high_risk_percentage']
                no_risk_percentage = 100 - risk_percentage
                labels = ["Having Lung Cancer - High Risk", "Not Having Lung Cancer - High Risk"]
            elif st.session_state['result'] == "Medium Risk":
                risk_percentage = st.session_state['medium_risk_percentage']
                no_risk_percentage = 100 - risk_percentage
                labels = ["Having Lung Cancer - Medium Risk", "Not Having Lung Cancer - Medium Risk"]
            else:  # Low Risk
                risk_percentage = st.session_state['low_risk_percentage']
                no_risk_percentage = 100 - risk_percentage
                labels = ["Having Lung Cancer - Low Risk", "Not Having Lung Cancer - Low Risk"]
    
            # Show the estimated risk percentage
            st.write(f"### Estimated Risk of Developing Lung Cancer: {risk_percentage:.2f}%")

            # Data for the pie chart
            sizes = [risk_percentage, no_risk_percentage]

            # Create and display the pie chart
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')  # Equal aspect ratio ensures that the pie chart is circular.
            st.pyplot(fig)
