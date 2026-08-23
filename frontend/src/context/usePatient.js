import { useContext } from "react";
import { PatientContext } from "./PatientContextInstance";

export function usePatient() {
  const context = useContext(PatientContext);
  if (!context) {
    throw new Error("usePatient must be used within a PatientProvider");
  }
  return context;
}
