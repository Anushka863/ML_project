function DiseaseCard({ icon, title, description }) {
  return (
    <div className="disease-card">

      <div className="disease-icon">
        {icon}
      </div>

      <h3>{title}</h3>

      <p>{description}</p>

    </div>
  );
}

export default DiseaseCard;