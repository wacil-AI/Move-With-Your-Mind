from Tri import inserer_dans_trie
# On importe la fonction d'insertion d'un élément à la bonne place
# dans un tableau trié

class File:
    """
    Classe file de priorité qui crée des instances file.

    Contient :
    - self.add(e) -> ajoute un élément e à la bonne place dans une
    file triée grâce à inserer_dans_trier
    - self.pop(e) -> supprime l'element du debut de la file et le renvoie, s'il existe
    - self.is_empty() -> renvoie True si la file est vide, False sinon
    """

    def __init__(self, values=[]):
        self.values = list(values)

    def is_empty(self):
        """
        Méthode qui renvoie True si la file est vide, False sinon.
        """
        return self.values == []
    
    def add(self, e):
        """
        Méthode qui ajoute l'element e au bon endroit dans une file triée.
        """
        self.values = inserer_dans_trie(self.values,e)
        
    def pop(self):
        """
        Méthode qui supprime l'element du debut de la file et le renvoie, s'il existe.
        """
        if not self.is_empty():
            return self.values.pop(0)

